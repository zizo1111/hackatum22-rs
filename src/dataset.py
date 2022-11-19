from __future__ import print_function, division
import os
import torch
import nibabel
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np

Z_IDX = 13

class MicrowaveDataset(Dataset):

    def __init__(self, root_dir, labels_path):
        self.root_dir = root_dir
        self.labels_path = labels_path

        # list all files in root directory   
        l=os.listdir(self.root_dir)
        self.file_names = list(set([x.split('.')[0] for x in l]))
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        pass

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