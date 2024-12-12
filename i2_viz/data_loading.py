import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import nibabel as nib
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker


run_ratings_dict = {
    'sub-01_ses-V1_task-S2_run-02_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold': 'PID1_v1_s2_r2 - 2023-09-01',
    'sub-01_ses-V1_task-S1_run-03_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold': 'PID1_v1_s1_r3 - 2023-09-01',
    # 'sub-01_ses-V1_task-S0_run-01_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold': 'PID1_v1_s0_r1 - 2023-08-31'
}
run_options = list(run_ratings_dict.keys())

parcel_labels = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.txt', sep='\t', names=['label', 'parcellation', 'x', 'y', 'z', 't'])['parcellation'].to_list()
parcel_labels_map = {v: k for k, v in dict(enumerate(parcel_labels, start=1)).items()}
default_parcels = [p for p in parcel_labels if "Default" in p]


def load_ratings(ratings_path):
    ratings_df = pd.read_csv(ratings_path)
    ratings_df.columns = ratings_df.columns.str.strip()
    ratings_df.Question = ratings_df.Question.str.strip()
    return ratings_df


class I2Run:
    # _cache = {}

    def __init__(self, run_prefix, window_size=44, step_size=2, frame_ms=900):
        self.run_prefix = run_prefix
        self.window_size = window_size
        self.step_size = step_size
        self.frame_ms = frame_ms
        
        self.load_data()


    def load_data(self):

        # if self.run_prefix in I2Run._cache:
        #     print(f"Loading {self.run_prefix} from cache.")
        #     cached_data = I2Run._cache[self.run_prefix]
        #     self.__dict__.update(cached_data)
        # else:
            print(f"Loading new data for {self.run_prefix}.")
            # Load image and ratings based on the current run_prefix
            self.run_img = nib.load(f'data/signal_intensity/{self.run_prefix}.nii.gz')
            self.ratings_prefix = run_ratings_dict[self.run_prefix]
            self.ratings_df = load_ratings(f'data/ratings/{self.ratings_prefix}.csv')

            # Load and resample parcellation image
            self.parcel_img_raw = nib.load('data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii')
            self.parcel_img_resampled = image.resample_img(self.parcel_img_raw, target_affine=self.parcel_img_raw.affine, target_shape=self.run_img.shape[:3])

            # Load parcellation info
            self.parcel_info_df = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.csv')
            self.n_parcels = len(self.parcel_info_df)

            # Load or create parcellated data
            if os.path.isfile(f'data/signal_intensity/{self.run_prefix}_parcellated.nii.gz') and os.path.isfile(f'data/signal_intensity/{self.run_prefix}_parcellated_timeseries.npy'):
                self.run_img_parcellated = nib.load(f'data/signal_intensity/{self.run_prefix}_parcellated.nii.gz')
                self.timeseries = np.load(f'data/signal_intensity/{self.run_prefix}_parcellated_timeseries.npy')
            else:
                print('Parcellated file does not exist, performing parcellation...')
                self.parcellate()

            # Functional connectivities
            self.calculate_functional_connectivities()

            # # Cache the instance data for future use
            # I2Run._cache[self.run_prefix] = self.__dict__.copy()
    
    
    def parcellate(self, out=True):
        '''Parcellate the target image according to Schaefer 100 parcellation'''
        masker = NiftiLabelsMasker(
            self.parcel_img_resampled,
            labels=['Background'] + list(self.parcel_info_df['full_name']),
            smoothing_fwhm=6,
            standardize='zscore_sample',
            memory='nilearn_cache',
        )
        self.timeseries = masker.fit_transform(self.run_img)
        self.run_img_parcellated = masker.inverse_transform(self.timeseries)

        if out:
            out_path = f'data/signal_intensity/{self.run_prefix}_parcellated.nii.gz'
            nib.save(self.run_img_parcellated, out_path)
            print(f'Saved parcellated file to {out_path}')

            out_path = f'data/signal_intensity/{self.run_prefix}_parcellated_timeseries.npy'
            np.save(out_path, self.timeseries)
            print(f'Saved timeseries to {out_path}')

    def calculate_functional_connectivities(self):
        
        self.calculate_windows()

        # dfc matrix
        if os.path.isfile(f'data/functional_connectivity/{self.run_prefix}_dynamic_fc.npy'):
            self.dfc_matrix = np.load(f'data/functional_connectivity/{self.run_prefix}_dynamic_fc.npy')
        else:
            print('Dynamic FC matrix does not exist, performing calculations...')
            self.dfc_matrix = self.dynamic_fuctional_connectivity_matrix()

        # dgfc matrix
        self.dgfc_matrix = self.dyamic_global_functional_connectivity()

        # dgfc volume
        if os.path.isfile(f'data/functional_connectivity/{self.run_prefix}_dynamic_gfc.nii.gz'):
            self.dgfc_img = nib.load(f'data/functional_connectivity/{self.run_prefix}_dynamic_gfc.nii.gz')
        else:
            print('Global functional connectivity volumes do not exist, performing calculations...')
            self.dgfc_img = self.dynamic_global_functional_connectivity_volumes()

        self.dgfc_vol = self.dgfc_img.get_fdata()

    def calculate_windows(self):
        self.n_windows = (self.run_img.shape[-1] - self.window_size) // self.step_size + 1
        self.window_timestamps = (np.arange(self.n_windows) * self.step_size + self.window_size // 2) * self.frame_ms / 1000


    def dynamic_fuctional_connectivity_matrix(self, out=True):
        '''
        For each window in a sliding window, calculate the functional connectivity matrix.
        The FC matrix is the correlation of activity between each pair of parcels in Schafer 100 parcels.
        Produces a 3D matrix of shape (n_windows, 100, 100) where n_windows is the number of windows.
        '''

        # Initialize dynamic correlation matrices
        dfc_matrix = np.zeros((self.n_windows, 100, 100))

        # Calculate connectivity matrix for each time window
        for i in range(self.n_windows):
            window_timeseries = self.timeseries[i * self.step_size:i * self.step_size + self.window_size, :]
            correlation_measure = ConnectivityMeasure(
                kind="correlation",
                standardize="zscore_sample",
            )
            fc = correlation_measure.fit_transform([window_timeseries])[0]
            np.fill_diagonal(fc, 0)
            dfc_matrix[i] = fc

        if out:
            out_path = f'data/functional_connectivity/{self.run_prefix}_dynamic_fc.npy'
            np.save(out_path, dfc_matrix)
            print(f'Saved dynamic FC matrix to {out_path}')

        return dfc_matrix

    def dyamic_global_functional_connectivity(self):
        '''
        Calculate the global functional connectivity of the brain for each window.
        The GFC is the average correlation of activity between each parcel and the rest of the brain.
        Produces a 1D array of shape (n_windows, n_parcels).
        '''
        return np.array([np.sum(W, axis=0) / (self.n_parcels - 1) for W in self.dfc_matrix])

    def dynamic_global_functional_connectivity_volumes(self, out=True):
        '''
        Maps the global functional connectivity to a 4D volume. Returns a nibabel image.
        '''

        # Create a template volume of parcel labels
        frame_template = self.parcel_img_resampled.get_fdata()
        x, y, z = frame_template.shape
        
        # initialize an empty array to store the volumes
        volumes = np.zeros((x, y, z, self.n_windows)) 

        for i in tqdm(range(self.n_windows), desc="Generating volumes"):
            frame_volume = frame_template.copy()
            frame_values = self.dynamic_gfc[i] # get the GFC values for the frame
            frame_values_map = dict(enumerate(frame_values, 1)) # make a map of parcels to GFC values

            # Replace the parcel labels in the template volume with the GFC values
            for parcel, gfc_value in frame_values_map.items():
                frame_volume[frame_volume == parcel] = gfc_value

            volumes[:, :, :, i] = frame_volume

        dgfc_img = nib.Nifti1Image(volumes, affine=self.parcel_img_resampled.affine)

        if out:
            out_path = f'data/functional_connectivity/{self.run_prefix}_dynamic_gfc.nii.gz'
            nib.save(dgfc_img, out_path)
            print(f'Saved global functional connectivity volumes to {out_path}')
        
        return dgfc_img