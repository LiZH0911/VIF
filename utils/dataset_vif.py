import os
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as utils


class Fusion_Dataset(data.Dataset):
    def __init__(self, dir_ir, dir_vi, pair_num=None):
        super(Fusion_Dataset, self).__init__()

        self.paths_ir, self.names_ir = utils.FileHandler.list_img_paths(dir_ir, num=pair_num)
        self.paths_vi, self.names_vi = utils.FileHandler.list_img_paths(dir_vi, num=pair_num)


    def __getitem__(self, index):

        path_ir = self.paths_ir[index]
        path_vi = self.paths_vi[index]
        name = self.names_ir[index]

        # read image
        img_ir = utils.FileHandler.imread_uint(path_ir, n_channels=1)
        img_vi = utils.FileHandler.imread_uint(path_vi, n_channels=3)

        # normalization
        img_ir = utils.FormatConversion.uint2single(img_ir)
        img_vi = utils.FormatConversion.uint2single(img_vi)

        # numpy to tensor, (h,w,c) to (c,h,w)
        img_ir = utils.FormatConversion.single2tensor3(img_ir)
        img_vi = utils.FormatConversion.single2tensor3(img_vi)

        # return {'img_ir': img_ir,
        #         'img_vi': img_vi,
        #         'path_ir': path_ir,
        #         'path_vi': path_vi,
        #         'name_ir': name_ir,
        #         'name_vi': name_vi}
        return img_ir, img_vi, name

    def __len__(self):
        return len(self.paths_ir)
