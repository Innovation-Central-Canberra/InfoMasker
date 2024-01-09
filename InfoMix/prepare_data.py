# This file is mainly based on prepare_data.py in **Libri2Mix** in [Speechbrain](https://speechbrain.github.io)
"""
The functions to create the .csv files for InfoMix

Author
 * Peng Huang 2023
"""

import os
import csv

def prepare_infomix(
    datapath, 
    savepath, 
    n_spks=1,
    skip_prep=False,
    fs=8000,
):
    """
    Prepare .csv files for infomix

    Arguments:
    ----------
        datapath (str) : path for the infomix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
    """
    
    if skip_prep:
        return
    create_infomix_csv(datapath, savepath, fs)

def create_infomix_csv(datapath,
    savepath,
    fs,
    n_spks=1,
    set_types = ['train_data_8k', 'test_data_8k']
):
    """
    This functions creates the .csv file for the infomix dataset
    """

    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, 'mixture/')
        s1_path = os.path.join(datapath, set_type, 'audio/')
        noise_ref_path = os.path.join(datapath, set_type, 'noise/')

        files = os.listdir(mix_path)
        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        noise_ref_fl_paths = [noise_ref_path + fl for fl in files]

        csv_columns = [
            "ID",
            "mix_wav",
            "noise_ref_wav",
            "s1_wav",
        ]

        with open(savepath+"/infomix_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            
            for i, (mix_path, noise_ref_path, s1_path) in enumerate(
                zip(mix_fl_paths, noise_ref_fl_paths, s1_fl_paths)
            ):
                row = {
                    "ID": i,
                    "mix_wav": mix_path,
                    "noise_ref_wav": noise_ref_path,
                    "s1_wav": s1_path
                }
                writer.writerow(row)