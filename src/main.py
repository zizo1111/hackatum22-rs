"""
HackaTUM 2022 - Rohde & Schwarz
"""

import os
import json
import math
import nibabel
import numpy as np
from matplotlib import pyplot as plt
from preproccess import *
from dataset import import_volume

from visualize import visualize_features

C0 = 299792458
FC = 77e9
LAMBDA = C0 / FC

Z_IDX = 0

def process_display(filename, path):
    volume, x_vec, y_vec, z_vec = import_volume(filename, path)
    
    Nx = x_vec.size
    Ny = y_vec.size

    kx = (np.arange(-Nx / 2, Nx / 2 - 1)) / ((Nx - 1) * np.diff(x_vec[:2]))
    ky = (np.arange(-Ny / 2, Ny / 2 - 1)) / ((Ny - 1) * np.diff(y_vec[:2]))

    kx_n = kx * LAMBDA
    ky_n = ky * LAMBDA

    volume_max, kmax = compute_mip(volume)

    counts = np.bincount(np.array(kmax.flat))
    counts[0] = 0
    counts[-1] = 0
    max_z = np.argmax(counts)

    volume_max_range = (np.min(np.abs(volume_max)), np.max(np.abs(volume_max)))
    alpha_data = np.clip(
        1.8 * ((np.abs(volume_max) - volume_max_range[0]) / (volume_max_range[1] - volume_max_range[0])) - 0.25, 0, 1,
    )

    # 1 --> visualize magnitude of the MIP
    image = 20 * np.log10(np.abs(volume_max / np.max(volume_max)))

    # 2 --> visualize phase of the MIP (opacity scaled by alpha_data)
    _, volume_max_phase = complex2magphase(
        np.multiply(volume_max, np.exp(((1j * 2 * math.pi) / LAMBDA) * 2 * z_vec[kmax]))
    )

    # 3 --> visualize the phase of a selected slice (opacity scaled by alpha_data)
    _, V_slice_phase = complex2magphase(volume[:, :, max_z])

    # 4 --> visualize the distance of the MIP (opacity scaled by alpha_data)

    # 5 --> visualize the 2D FFT of the MIP
    S_MIP = compute_fft(volume_max)
    S_MIP_mag_dB = 20 * np.log10(np.abs(S_MIP))

    # 6 --> visualize the 2D FFT of a single slice
    S_slice = compute_fft(volume[:, :, max_z])
    S_slice_mag_dB = 20 * np.log10(np.abs(S_slice))
  

    vis_dict = {
        'x_vec' : x_vec,
        'y_vec' : y_vec,
        'z_vec' : z_vec,
        'alpha_data' : alpha_data,
        'max_z': max_z
    }
    np_img = visualize_features([image, 180 / math.pi * volume_max_phase, 180 / math.pi * V_slice_phase, z_vec[kmax],
       S_MIP_mag_dB, S_slice_mag_dB], vis_dict, filename)


def run_folder(root_dir):

    # list all files in root directory   
    l = os.listdir(root_dir)

    files = []
    for x in l:
        split = x.split('.')
        if(split[1] != 'png'):
            files.append(split[0])
    
    file_names = list(set(files))

    for file_name in file_names:
        process_display(file_name, root_dir)
        plt.close()

if __name__ == "__main__":
   
    # process_display(
    #     r"example-1", r"../examples/"
    #     #r"20221119-150759-488_reco", r"/media/hdd_4tb/Datasets/rohde_and_schwarz_measurements/"
    #     )


    # plt.show()  

    run_folder(r"/media/hdd_4tb/Datasets/rohde_and_schwarz_measurements/")