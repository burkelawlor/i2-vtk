import pandas as pd
import numpy as np
import glob as glob

import nibabel as nib
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

import matplotlib.pyplot as plt
import seaborn as sns

def parcellate(target_nii, parcel_nii, parcel_labels, out=True):
    # Load files
    target_raw = nib.load(target_nii)
    parcel_raw = nib.load(parcel_nii)
    
    # For the parcellation img, resample the shape to the target img, but not the affine (to preserve the network labels)
    parcel_resampled = image.resample_img(parcel_raw, target_affine=parcel_raw.affine, target_shape=target_raw.shape[:3]) 

    # Apply the parcellation masker
    masker = NiftiLabelsMasker(
        parcel_resampled,
        labels = parcel_labels,
        smoothing_fwhm=6,
        standardize='zscore_sample',
        standardize_confounds="zscore_sample",
        memory='nilearn_cache',
    )
    timeseries = masker.fit_transform(target_raw)
    target_parcellated = masker.inverse_transform(timeseries)

    if out:
        out_file = target_nii.replace('.nii', '_parcellated.nii')
        nib.save(target_parcellated, out_file)
        print(f'Saved parcellated file to {out_file}')

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
    def __init__(self, subject, session, task, run):
        self.target_nii = f'data/fmri/sub-0{subject}_ses-V{session}_task-S{task}_run-0{run}_space-MNI152NLin2009cAsym_res-2_desc-denoisedSmoothed_bold.nii.gz'
        parcel_nii = 'data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'
        
        parcel_df = pd.read_csv('data/Schaefer2018_100Parcels_7Networks_order.csv')
        parcel_labels = ['Background'] + list(parcel_df['full_name'])

        self.target_parcellated, self.timeseries = parcellate(self.target_nii, parcel_nii, parcel_labels, out=False)
        self.dfc_matrix, self.timestamps = dynamic_fc(self.timeseries, window_size=44, step_size=2, frame_ms=900)
        self.gdfc_matrix = dynamic_gfc(self.dfc_matrix)

        ratings_file = glob.glob(f'data/ratings/PID{subject}_v{session}_s{task}_r{run} - *.csv')[0]    
        self.ratings_df = pd.read_csv(ratings_file)

    def i2_plot_agg_gdfc_with_ratings(self, parcels_of_interest):
        plot_agg_gdfc_with_ratings(self.timestamps, self.gdfc_matrix, self.parcel_labels, parcels_of_interest, self.ratings_df)

i2_run = I2Run(subject=1, session=1, task=2, run=2)

parcels_of_interest = ['7Networks_LH_Default_PCC_1', '7Networks_LH_Default_PCC_2', '7Networks_RH_Default_PFCm_1', '7Networks_RH_Default_PFCm_2', '7Networks_RH_Default_PFCm_3', '7Networks_RH_Default_PCC_1', '7Networks_RH_Default_PCC_2']

plt.figure(figsize=(16,6))
i2_run.i2_plot_agg_gdfc_with_ratings(parcels_of_interest)
plt.show()