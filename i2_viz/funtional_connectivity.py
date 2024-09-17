import pandas as pd
import numpy as np
import glob as glob
import os
import time

import nibabel as nib
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

import matplotlib.pyplot as plt
import seaborn as sns


def parcellate(target_img, parcel_img, parcel_labels, out=None):
    
    # For the parcellation img, resample the shape to the target img, but not the affine (to preserve the network labels)
    parcel_resampled = image.resample_img(parcel_img, target_affine=parcel_img.affine, target_shape=target_img.shape[:3]) 

    # Apply the parcellation masker
    masker = NiftiLabelsMasker(
        parcel_resampled,
        labels = parcel_labels,
        smoothing_fwhm=6,
        standardize='zscore_sample',
        standardize_confounds="zscore_sample",
        memory='nilearn_cache',
    )
    timeseries = masker.fit_transform(target_img)
    target_parcellated = masker.inverse_transform(timeseries)

    if out:
        nib.save(target_parcellated, out)
        print(f'Saved parcellated file to {out}')

    return target_parcellated, timeseries



def stationary_fc(timeseries):
    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    sfc_matrix = correlation_measure.fit_transform([timeseries])[0]
    np.fill_diagonal(sfc_matrix, 0)
    return sfc_matrix



def dynamic_fc(timeseries, window_size=44, step_size=2, frame_ms=900):
    # Set number of windows and calculate timestamp of center frame
    n_windows = (timeseries.shape[0] - window_size) // step_size + 1

    # Initialize dynamic correlation matrices
    dfc_matrix = np.zeros((n_windows,100,100))
    window_timestamps = np.zeros(n_windows)

    # Calculate connectivity matrix for each time window
    for i in range(n_windows):
        window_timeseries = timeseries[i*step_size:i*step_size+window_size, :]
        dfc_matrix[i] = stationary_fc(window_timeseries)
        window_timestamps[i] = (i * step_size + window_size // 2) * frame_ms / 1000

    return dfc_matrix, window_timestamps


def dynamic_gfc(dfc_matrix):
    n_parcels = dfc_matrix.shape[-1]
    dgfc = np.array([np.sum(W, axis=0) / (n_parcels-1) for W in dfc_matrix])
    return dgfc


def plot_dfc(timestamps, fc_array):
    ax = sns.lineplot(x=timestamps, y=fc_array)
    plt.xlabel('time (sec)')
    plt.ylabel('functional connectivity')
    return ax


def plot_agg_gdfc(timestamps, gdfc_matrix, parcel_labels, parcels_of_interest):
    parcel_indices = [parcel_labels.index(label) - 1 for label in parcels_of_interest] # minus 1 to account for the background label
    agg_fc = gdfc_matrix[:, parcel_indices].mean(axis=1)
    return plot_dfc(timestamps, agg_fc)


def plot_agg_gdfc_with_ratings(timestamps, gdfc_matrix, parcel_labels, parcels_of_interest, ratings_df):
    ax = plot_agg_gdfc(timestamps, gdfc_matrix, parcel_labels, parcels_of_interest)

    qs_of_interest = [' Positive/Negative', ' Acceptance/Resistance', ' No Insight/Strong Insight']
    ratings_to_plot = ratings_df[ratings_df[' Question'].isin(qs_of_interest)]
    
    ax2 = ax.twinx()
    sns.scatterplot(x=' Seconds since start', y=' Answer', hue=' Question', data=ratings_to_plot, ax=ax2, palette=['red','green','blue'])
    ax2.set_ylabel('subjective ratings', rotation=270, labelpad=15)
    ax2.set_ylim(0, 4.5)
    ax2.set_yticks(np.arange(0, 5, step=1))
    


# target_nii = 'data/fmri/sub-01_ses-V1_task-S2_run-02_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'
# parcel_nii = 'data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'
# parcel_df = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.csv')
# parcel_labels = ['Background'] + list(parcel_df['full_name'])

# target_parcellated, timeseries = parcellate(target_nii, parcel_nii, parcel_labels)
# dfc_matrix, timestamps = dynamic_fc(timeseries)
# gdfc_matrix = dynamic_gfc(dfc_matrix) 

# # parcels_of_interest = parcel_df.loc[parcel_df['network'] == 'Default', 'full_name'].to_list()
# parcels_of_interest = ['7Networks_LH_Default_PCC_1', '7Networks_LH_Default_PCC_2', '7Networks_RH_Default_PFCm_1', '7Networks_RH_Default_PFCm_2', '7Networks_RH_Default_PFCm_3', '7Networks_RH_Default_PCC_1', '7Networks_RH_Default_PCC_2']
# ratings_df = pd.read_csv('data/ratings/PID1_v1_s2_r2 - 2023-09-01.csv')

# plot_agg_gdfc_with_ratings(timestamps, gdfc_matrix, parcels_of_interest, ratings_df)
# plt.show()


# subject, session, task, run = 1, 1, 2, 2
# import os
# os.path.isfile(f'data/fmri/sub-0{subject}_ses-V{session}_task-S{task}_run-0{run}_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz')



class I2Run():
    def __init__(self, target_nii, target_ratings):
        self.target_nii = target_nii
        self.target_img = nib.load(self.target_nii)

        self.target_ratings = target_ratings
        self.ratings_df = pd.read_csv(self.target_ratings)

    def parcellate_i2(self, parcel_nii, parcel_labels):
        self.parcel_nii = parcel_nii
        self.parcel_labels = parcel_labels
        self.parcel_img = nib.load(self.parcel_nii)
        self.target_parcellated, self.timeseries = parcellate(self.target_img, self.parcel_img, self.parcel_labels)
    
    def dynamic_fc_i2(self, window_size=44, step_size=2, frame_ms=900):
        self.dfc_matrix, self.timestamps = dynamic_fc(self.timeseries, window_size, step_size, frame_ms)

    def dynamic_gfc_i2(self):
        self.gdfc_matrix = dynamic_gfc(self.dfc_matrix)

    def i2_plot_agg_gdfc_with_ratings(self, parcels_of_interest):
        plot_agg_gdfc_with_ratings(self.timestamps, self.gdfc_matrix, self.parcel_labels, parcels_of_interest, self.ratings_df)

    def gfc_volumes(self, out=False):

        # Crate a template volume of parcel labels
        frame_template = self.parcel_img.get_fdata()
        x, y, z = frame_template.shape
        n_frames = self.gdfc_matrix.shape[0]

        volumes = np.zeros((x, y, z, n_frames)) # initialize an empty array to store the volumes
        for i in range(n_frames):
            
            frame_volume = frame_template.copy() # make a copy of the template volume
            frame_values = i2_run.gdfc_matrix[i] # get the GFC values for the frame
            frame_values_map = dict(enumerate(frame_values, 1)) # make a map of parcels to GFC values

            # Replace the parcel labels in the template volume with the GFC values
            for parcel, gfc_value in frame_values_map.items():
                frame_volume[frame_volume == parcel] = gfc_value
        
            volumes[:,:,:,i] = frame_volume

        self.gfc_volumes = volumes

        # Save the volumes to a file
        if out:
            np.save(out, volumes)


parcel_nii = 'data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'
parcel_df = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.csv')
parcel_labels = ['Background'] + list(parcel_df['full_name'])
parcels_of_interest = ['7Networks_LH_Default_PCC_1', '7Networks_LH_Default_PCC_2', '7Networks_RH_Default_PFCm_1', '7Networks_RH_Default_PFCm_2', '7Networks_RH_Default_PFCm_3', '7Networks_RH_Default_PCC_1', '7Networks_RH_Default_PCC_2']

target_nii = f'data/fmri/sub-0{1}_ses-V{1}_task-S{2}_run-0{2}_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'
target_ratings = f'data/ratings/PID{1}_v{1}_s{2}_r{2} - 2023-09-01.csv'

i2_run = I2Run(target_nii, target_ratings)
i2_run.parcellate_i2(parcel_nii, parcel_labels)
i2_run.dynamic_fc_i2()
i2_run.dynamic_gfc_i2()
i2_run.gfc_volumes(out=f'data/gfc_volumes/sub-0{1}_ses-V{1}_task-S{2}_run-0{2}_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.npy')

# i2_run.i2_plot_agg_gdfc_with_ratings(parcels_of_interest)
# plt.show()

# # Sanity checks
# i2_run.gfc_volumes
# i2_run.gfc_volumes.shape

# # Plot a single volume (as a check that we did it right)
# from nilearn import plotting
# frame_img = nib.Nifti1Image(i2_run.gfc_volumes[:,:,:,0], affine=i2_run.parcel_img.affine)
# display = plotting.plot_stat_map(
#     frame_img, 
#     vmin=-0.3, 
#     vmax=0.3, 
#     cmap='cold_hot',
#     display_mode='x', 
#     cut_coords=[4], 
#     draw_cross=False)
# plotting.show()


