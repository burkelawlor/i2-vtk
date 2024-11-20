import os
import numpy as np
import pandas as pd
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import seaborn as sns

import nibabel as nib
from nilearn import image, plotting, masking
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

parcellation_file = 'data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'
parcellation_labels = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.txt', sep='\t', names=['label', 'parcellation', 'x', 'y', 'z', 't'])['parcellation'].to_list()

class fmriFunctionalConnectivity():
    def __init__(self, fmri_file, parcellation_file, parcellation_labels):
        self.fmri_file = fmri_file
        self.fmri_img = nib.load(self.fmri_file)
        
        self.parcellation_file = parcellation_file
        self.parcellation_img_raw = nib.load(parcellation_file)
        self.parcellation_img = image.resample_img(self.parcellation_img_raw, target_affine=self.parcellation_img_raw.affine, target_shape=self.fmri_img.shape[:3]) # Resample the shape but not the affine to preserve the network labels

        self.parcellation_labels = parcellation_labels
    
    def apply_parcellation_masker(self, out_file=None):
        labels = self.parcellation_labels.copy()
        labels.insert(0, 'Background')

        masker = NiftiLabelsMasker(
            self.parcellation_img,
            labels = labels,
            smoothing_fwhm=6,
            standardize='zscore_sample',
            standardize_confounds="zscore_sample",
            memory='nilearn_cache',
        )
        self.timeseries = masker.fit_transform(self.fmri_file)
        self.fmri_parcellated = masker.inverse_transform(self.timeseries)

        if out_file:
            nib.save(self.fmri_parcellated, out_file)


    # def calculate_dynamic_correlation_matrix(self, window_size=44, step_size=2, out_file=None):
    #     # Set number of windows
    #     n_windows = (self.timeseries.shape[0] - window_size) // step_size + 1
    #     print(f'Number of windows: {n_windows}')

    #     # Calculate connectivity matrix for each time window
    #     dynamic_correlation_matrix = np.zeros((n_windows,100,100))
    #     for i in range(n_windows):
    #         window_fdata = self.timeseries[i*step_size:i*step_size+window_size, :]
    #         connectivity_measure = ConnectivityMeasure(
    #             kind="correlation",
    #             standardize="zscore_sample",
    #         )
    #         window_correlation_matrix = connectivity_measure.fit_transform([window_fdata])[0]
    #         np.fill_diagonal(window_correlation_matrix, 0)
    #         dynamic_correlation_matrix[i] = window_correlation_matrix
        
    #     if out_file:
    #         print(f'Saving to {out_file}')
    #         np.save(out_file, dynamic_correlation_matrix)
        
    #     self.dynamic_correlation_matrix = dynamic_correlation_matrix

    def calculate_dynamic_correlation_matrix(self, window_size=44, step_size=2, out_file=None):
        # Set number of windows
        n_windows = (self.timeseries.shape[0] - window_size) // step_size + 1
        print(f'Number of windows: {n_windows}')

        # Calculate connectivity matrix for each time window
        dynamic_correlation_matrix = np.zeros((n_windows, 100, 100))
        dynamic_correlation_matrix_timestamps = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_fdata = self.timeseries[i*step_size:i*step_size+window_size, :]
            connectivity_measure = ConnectivityMeasure(
                kind="correlation",
                standardize="zscore_sample",
            )
            window_correlation_matrix = connectivity_measure.fit_transform([window_fdata])[0]
            np.fill_diagonal(window_correlation_matrix, 0)
            dynamic_correlation_matrix[i] = window_correlation_matrix
            
            # Calculate the timestamp for the center of the window
            center_frame = i * step_size + window_size // 2
            timestamp = center_frame * 0.9  # Each frame is 900ms apart
            dynamic_correlation_matrix_timestamps[i] = timestamp

        if out_file:
            print(f'Saving to {out_file}')
            np.save(out_file, dynamic_correlation_matrix)
        
        self.dynamic_correlation_matrix = dynamic_correlation_matrix
        self.dynamic_correlation_matrix_timestamps = dynamic_correlation_matrix_timestamps

    def connectivity_matrices_to_images(self, out_dir='images_static/corr_matrices', dpi=100):
        for i in range(self.dynamic_correlation_matrix.shape[0]):
            correlation_matrix = self.dynamic_correlation_matrix[i]
            plotting.plot_matrix(
                correlation_matrix,
                figure=(12,12),
                labels=self.labels, # The labels we have start with the background (0), hence we skip the first label
                vmax=0.8,
                vmin=-0.8,
                title="Correlation matrix of Schaefer 100 parcellation",
                reorder=False,
            )
            plt.savefig(os.path.join(out_dir, f'corr_matrix_{i}.png', dpi=dpi))
    
    def calculate_gfc(self):
        def calculate_per_parcel_gfc(correlation_matrix):
            '''Calculate the global functional connectivity for each parcel in the Schaefer 100 parcellation
                aka the average correlation of each parcel to all other parcels'''
            n_parcels, _ = correlation_matrix.shape
            per_parcel_connectivity = np.sum(correlation_matrix, axis=0) / (n_parcels-1) # Divide by n_parcels-1 to exclude the diagonal
            return per_parcel_connectivity
        
        per_parcel_connectivities = np.array([calculate_per_parcel_gfc(cm) for cm in self.dynamic_correlation_matrix])
        self.per_parcel_connectivities = per_parcel_connectivities


# sample = 'sub-01_ses-V1_task-S0_run-01_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold'
# fc_bl = fmriFunctionalConnectivity(f'data/{sample}/{sample}.nii.gz', parcellation_file, parcellation_labels)
# fc_bl.apply_parcellation_masker()
# fc_bl.calculate_dynamic_correlation_matrix(out_file=f'data/{sample}/dynamic_correlation_matrix.npy')
# fc_bl.calculate_gfc()

# sample = 'sub-01_ses-V1_task-S2_run-02_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold'
# fc_drug = fmriFunctionalConnectivity(f'data/{sample}/{sample}.nii.gz', parcellation_file, parcellation_labels)
# fc_drug.apply_parcellation_masker()
# fc_drug.calculate_dynamic_correlation_matrix(out_file=f'data/{sample}/dynamic_correlation_matrix.npy')
# fc_drug.calculate_gfc()