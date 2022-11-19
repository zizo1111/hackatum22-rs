from __future__ import print_function, division
import os
import torch
import nibabel
import json
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preproccess import *

Z_IDX = 13
C0 = 299792458
FC = 77e9
LAMBDA = C0 / FC


class MicrowaveDataset(Dataset):

    def __init__(self, root_dir, labels_path=None):
        self.root_dir = root_dir
        self.labels_path = labels_path

        # list all files in root directory   
        l=os.listdir(self.root_dir)
        self.file_names = list(set([x.split('.')[0] for x in l]))
        self.labels = None
        if labels_path is not None:
            self.labels = {}
            with open(labels_path, "r") as f:
                labels_dict = json.load(f)
                for values in labels_dict:
                    p_pivot = np.array(values['p_pivot'])
                    angles = np.array([values['alpha'], values['beta'], values['gamma']])

                    self.labels[values['file']] = np.hstack((p_pivot, angles))
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        volume, x_vec, y_vec, z_vec = self.import_volume(self.file_names[idx])

        volume_max, kmax = compute_mip(volume)
        volume_max_range = (np.min(np.abs(volume_max)), np.max(np.abs(volume_max)))
        
        # we need to scale by the alpha data
        alpha_data = np.clip(
            1.8 * ((np.abs(volume_max) - volume_max_range[0]) / (volume_max_range[1] - volume_max_range[0])) - 0.25, 0, 1,
        )

        # magnitude of the MIP
        mip_mag = 20 * np.log10(np.abs(volume_max / np.max(volume_max)))
        mip_mag = torch.from_numpy(mip_mag)

        # phase of the MIP
        _, volume_max_phase = complex2magphase(
            np.multiply(volume_max, np.exp(((1j * 2 * math.pi) / LAMBDA) * 2 * z_vec[kmax]))
        )
        volume_max_phase = torch.from_numpy(to_degrees(volume_max_phase))

    
        # the phase of a selected slice 
        _, V_slice_phase = complex2magphase(volume[:, :, Z_IDX - 1])
        V_slice_phase = torch.from_numpy(to_degrees(V_slice_phase))

        # The distance of the MIP
        mip_distance = z_vec[kmax]
        mip_distance = torch.from_numpy(mip_distance)

        # The 2D FFT of the MIP
        S_MIP = compute_fft(volume_max)
        S_MIP_mag_dB = 20 * np.log10(np.abs(S_MIP))
        S_MIP_mag_dB = torch.from_numpy(S_MIP_mag_dB)

        # 2D FFT of a single slice
        S_slice = compute_fft(volume[:, :, Z_IDX - 1])
        S_slice_mag_dB = 20 * np.log10(np.abs(S_slice))
        S_slice_mag_dB = torch.from_numpy(S_slice_mag_dB)

        preprocessed_input = torch.stack((mip_mag, volume_max_phase, V_slice_phase, mip_distance, S_MIP_mag_dB, S_slice_mag_dB))

        label = None
        if self.labels is not None:
            label = torch.from_numpy(self.labels[self.file_names[idx]])

        data = {'inputs': preprocessed_input, 
                'label': label}

        return data



    def import_volume(self, filename):
        """Import 3D volumetric data from file.

        Args:
            file_path (basestring): Absolute path for .img, .hdr or .json file.

        Returns:
            The volume definition given as the coordinate vectors in x, y, and z-direction.
        """

        path = self.root_dir

        file_path_img = os.path.join(path, f"{filename}.img")
        file_path_hdr = os.path.join(path, f"{filename}.hdr")
        file_path_json = os.path.join(path, f"{filename}.json")

        if not os.path.exists(file_path_img):
            raise Exception(f"Does not exist file: {file_path_img}")
        if not os.path.exists(file_path_hdr):
            raise Exception(f"Does not exist file: {file_path_hdr}")
        if not os.path.exists(file_path_json):
            raise Exception(f"Does not exist file: {file_path_json}")

        v_mag_phase = nibabel.load(file_path_hdr)
        _volume = v_mag_phase.dataobj[:, :, : v_mag_phase.shape[2] // 2] * np.exp(
            1j * v_mag_phase.dataobj[:, :, v_mag_phase.shape[2] // 2 :]
        )
        if len(_volume.shape) > 3:
            _volume = np.squeeze(_volume)

        with open(file_path_json, "rb") as vol_definition_file:
            def_json = json.load(vol_definition_file)
            _x_vec = def_json["origin"]["x"] + np.arange(def_json["dimensions"]["x"]) * def_json["spacing"]["x"]
            _y_vec = def_json["origin"]["y"] + np.arange(def_json["dimensions"]["y"]) * def_json["spacing"]["y"]
            _z_vec = def_json["origin"]["z"] + np.arange(def_json["dimensions"]["z"]) * def_json["spacing"]["z"]

            return _volume, _x_vec, _y_vec, _z_vec


if __name__ == "__main__":
    dataset = MicrowaveDataset('../data/dummy_measurements/volumes', '../data/dummy_measurements/labels_transformed.json')

    for i in range(len(dataset)):
        ret = dataset.__getitem__(i)
        print(ret['label'], ret['inputs'].shape)