import os
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as utils


class Dataset(data.Dataset):
    def __init__(self, dir_A, dir_B, in_channels):
        super(Dataset, self).__init__()
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A, self.names_A = utils.FileHandler.list_img_paths(dir_A)
        self.paths_B, self.names_B = utils.FileHandler.list_img_paths(dir_B)
        self.in_channels = in_channels

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        path_A = self.paths_A[index]
        path_B = self.paths_B[index]
        name_A = self.names_A[index]
        name_B = self.names_B[index]
        img_A = utils.FileHandler.imread_uint(path_A, self.in_channels)
        img_B = utils.FileHandler.imread_uint(path_B, self.in_channels)
       
        """
        # --------------------------------
        # get testing image pairs
        # --------------------------------
        """
        img_A = utils.FormatConversion.uint2single(img_A)
        img_B = utils.FormatConversion.uint2single(img_B)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = utils.FormatConversion.single2tensor3(img_A)
        img_B = utils.FormatConversion.single2tensor3(img_B)

        return {'img_A': img_A,
                'img_B': img_B,
                'path_A': path_A,
                'path_B': path_B,
                'name_A': name_A,
                'name_B': name_B}

    def __len__(self):
        return len(self.paths_A)
