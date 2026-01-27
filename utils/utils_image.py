import numpy as np
import torch
import torch.nn as nn
import os
import logging
import cv2
import time
import random
from natsort import natsorted
from PIL import Image

# logging.basicConfig(filename='program.log', filemode='w', level=logging.DEBUG)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------
class FileHandler:
    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                logger.error("Create store path failed on init: %s", e)

    # list original image paths
    @staticmethod
    def list_img_paths(img_dir, num=None):
        '''
        列出指定目录下的前num个图像文件路径
        '''
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'} # 常见的图像文件扩展名
        files = natsorted(os.listdir(img_dir)) # 读取文件并自然排序
        img_paths = [] # 列表存储所有图像的完整路径
        img_names = [] # 列表存储所有图像的文件名（无扩展名）
        index = 0
        for file in files:
            file_ext = os.path.splitext(file)[1].lower() # 读取扩展名
            if file_ext in extensions:
                img_path = os.path.join(img_dir, file)  # 拼接完整路径
                img_paths.append(img_path)
                name = file.split('.')  # 按点分割文件名（处理扩展名）
                img_names.append(name[0])
                index += 1
            if num is not None:
                if index >= num:
                    break
        return img_paths, img_names

    @staticmethod
    def imread_uint(path, n_channels=1):
        '''
        n_channels = 1 or 3
        (h,w,c)
        '''
        if n_channels == 1 :
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # HxW
            img = np.expand_dims(img, axis=2)  # HxWx1
        elif n_channels == 3:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # BGR or G
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # GGG HxWx3
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB HxWx3
        return img

    @staticmethod
    def save_img(img, output_path):
        '''将三维张量转换为图像并保存'''
        cv2.imwrite(output_path, img)

# ------------------------------------------------------------------------------------------------------------
class FormatConversion:
    '''
    numpy(single) <--->  numpy(uint)
    numpy(uint)   <--->  tensor
    numpy(single) <--->  tensor
    '''
    # --------------------------------------------
    # numpy(uint)  <--->  numpy(single)
    # --------------------------------------------

    @staticmethod
    def uint2single(img):
        return np.float32(img / 255.)

    @staticmethod
    def single2uint(img):
        return np.uint8((img.clip(0, 1)*255.).round())

    # --------------------------------------------
    # numpy(uint) (h, w, c) or (h, w)  <--->  tensor
    # --------------------------------------------

    # convert uint to 3-dimensional torch tensor
    @staticmethod
    def uint2tensor3(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

    # convert uint to 4-dimensional torch tensor
    @staticmethod
    def uint2tensor4(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

    # convert 2/3/4-dimensional torch tensor to uint
    @staticmethod
    def tensor2uint(tensor):
        img = tensor.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img*255.0).round())

    # --------------------------------------------
    # numpy(single) (h, w, c) or (h, w)  <--->  tensor
    # --------------------------------------------

    # convert single to 3-dimensional torch tensor
    @staticmethod
    def single2tensor3(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

    # convert single to 4-dimensional torch tensor
    @staticmethod
    def single2tensor4(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

    # convert torch tensor to single
    @staticmethod
    def tensor2single(img):
        img = img.data.squeeze().float().cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return img


    @staticmethod
    def save_tensor2uint(tensor, output_path):
        '''将三维张量转换为图像并保存'''
        img = FormatConversion.tensor2uint(tensor)
        cv2.imwrite(output_path, img)
        return img

# ------------------------------------------------------------------------------------------------------------
class Augmentation:

    @staticmethod
    def augment_img(img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    @staticmethod
    def augment_img_tensor4(img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return img.rot90(1, [2, 3]).flip([2])
        elif mode == 2:
            return img.flip([2])
        elif mode == 3:
            return img.rot90(3, [2, 3])
        elif mode == 4:
            return img.rot90(2, [2, 3]).flip([2])
        elif mode == 5:
            return img.rot90(1, [2, 3])
        elif mode == 6:
            return img.rot90(2, [2, 3])
        elif mode == 7:
            return img.rot90(3, [2, 3]).flip([2])

# ------------------------------------------------------------------------------------------------------------
class ImageChannelConversion:
    @staticmethod
    def rgb2ycbcr(img, only_y=True):
        '''same as matlab rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                  [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)

    @staticmethod
    def bgr2ycbcr(img, only_y=True):
        '''bgr version of rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                                  [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)

    @staticmethod
    def ycbcr2rgb(img):
        '''same as matlab ycbcr2rgb
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img = img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
        rlt = np.clip(rlt, 0, 255)
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)

    @staticmethod
    def channel_convert(in_c, tar_type, img_list):
        # conversion among BGR, gray and y
        if in_c == 3 and tar_type == 'gray':  # BGR to gray
            gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
            return [np.expand_dims(img, axis=2) for img in gray_list]
        elif in_c == 3 and tar_type == 'y':  # BGR to y
            y_list = [ImageChannelConversion.bgr2ycbcr(img, only_y=True) for img in img_list]
            return [np.expand_dims(img, axis=2) for img in y_list]
        elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
            return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
        else:
            return img_list

# ------------------------------------------------------------------------------------------------------------
class ImageProcess:
    @staticmethod
    def img_padding(img, c_h, c_w):
        '''
        确保图像尺寸能被指定的块尺寸整除，输入图像的 shape 要求为 (h, w, c)
        (h, w, c)图像 -> (h_patches * c_h, w_patches * c_w, c)
        '''
        h, w, c = img.shape[:]
        h_patches = int(np.ceil(h / c_h)) # 高度方向需要的分块数，np.ceil为向上取整函数
        w_patches = int(np.ceil(w / c_w)) # 宽度方向需要的分块数

        h_padding = h_patches * c_h - h # 高度方向需要填充的像素数
        w_padding = w_patches * c_w - w # 宽度方向需要填充的像素数
        # reflect, symmetric, wrap, edge, linear_ramp, maximum, mean, median, minimum
        img = np.pad(img, ((0, h_padding), (0, w_padding), (0, 0)), 'reflect') # 反射填充
        return img, [h_patches, w_patches, h_padding, w_padding]

    @staticmethod
    def img_patching(img, c_h, c_w):
        '''
        图像分块函数，(h, w, c)图像 -> (h_patches * w_patches, c_h, c_w, c)子图像
        '''
        img, pad_para = ImageProcess.img_padding(img, c_h, c_w)
        h, w, c = img.shape[:]
        h_patches = pad_para[0]
        w_patches = pad_para[1]

        patches = img.reshape(h_patches, c_h, w_patches, c_w, c) # 重塑为 (h_patches, c_h, w_patches, c_w, c)
        patches = patches.transpose(0, 2, 1, 3, 4) # 调整维度顺序为 (h_patches, w_patches, c_h, c_w, c)
        patch_matrix = patches.reshape(-1, c_h, c_w, c) # 最终重塑为 (h_patches * w_patches, c_h, c_w, c)

        return patch_matrix, pad_para

# ------------------------------------------------------------------------------------------------------------
# class DataLoader:
#     # load image paths of one epoch from original image paths
#     @staticmethod
#     def load_epoch_img_paths(img_paths, BATCH_SIZE, num_imgs=None, shuffle=True):
#         if num_imgs is None:
#             num_imgs = len(img_paths)
#         img_paths_epoch = img_paths[:num_imgs]
#         # random
#         if shuffle:
#             random.shuffle(img_paths_epoch)
#         mod = num_imgs % BATCH_SIZE
#         print('BATCH SIZE %d.' % BATCH_SIZE)
#         print('Train images number %d.' % num_imgs)
#         print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
#
#         if mod > 0:
#             print('Train set has been trimmed %d samples...\n' % mod)
#             img_paths_epoch = img_paths_epoch[:-mod]
#         num_batches = int(len(img_paths_epoch) // BATCH_SIZE)
#         return img_paths_epoch, num_batches
#
#     # get uint8 image of size HxWxC (RGB) from the path
#     @staticmethod
#     def imread_uint(path, n_channels=1):
#         '''
#         n_channels = 1 or 3
#         '''
#         if n_channels == 1 :
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # HxW
#             img = np.expand_dims(img, axis=2)  # HxWx1
#         elif n_channels == 3:
#             img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # BGR or G
#             if img.ndim == 2:
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # GGG HxWx3
#             elif img.ndim == 3:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB HxWx3
#         return img
#
#     # get one tensor(n, c, h, w) from a batch of image paths
#     @staticmethod
#     def imread_tensor4(paths, height=256, width=256, n_channels=1):
#         '''
#         以 RGB 或灰度格式读取多个路径的图像，resize 为指定 shape，并转换为 torch 张量
#         '''
#         if isinstance(paths, str):
#             paths = [paths]
#         imgs_list = [] # 用列表存储多个图像numpy数组
#         for path in paths:
#             img = DataLoader.imread_uint(path, n_channels)
#             img = cv2.resize(img, (height, width))
#             if n_channels == 1:
#                 img = np.reshape(img, [1, img.shape[0], img.shape[1]])  # (h, w) -> (c, h, w)
#             elif n_channels == 3:
#                 img = np.transpose(img, (2, 0, 1))  # (h, w, c) -> (c, h, w)
#             imgs_list.append(img)
#         # images_tensor = np.stack(images_list, axis=0) # 将多个numpy数组沿维度0堆叠, n个(c, h, w) -> (n, c, h, w)
#         # images_tensor = torch.from_numpy(images).float()
#         tensors = [torch.from_numpy(img) for img in imgs_list]
#         imgs_tensor = torch.stack(tensors, dim=0).float()
#         return imgs_tensor




# # 使用示例
# if __name__ == "__main__":
    # img = np.array([[
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16]
    # ]])
    # print("原始形状:", img.shape)  # 输出: (1, 4, 4)
    # img_tiled = np.tile(img, (3, 1, 1))
    # print("复制后形状:", img_tiled.shape)  # 输出: (3, 4, 4)
    # img_tiled = np.transpose(img_tiled,(1, 2, 0)).astype('uint8')
    # print("排列后形状:", img_tiled.shape)  # 输出: (4, 4, 3)
    #
    # path = os.path.join(os.getcwd(),"output/annotated_01_DenseFuse.png")
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    #
    # test_img = np.random.rand(3, 512, 512)
    # block_size = 256
    #--------------------------------
    # # 测试原版本
    # start = time.time()
    # patches1, para1 = crop_op(test_img.copy(), block_size, block_size)
    # time1 = time.time() - start
    #
    # # 测试优化版本
    # start = time.time()
    # patches2, para2 = img_patching(test_img.copy(), block_size, block_size)
    # time2 = time.time() - start
    #
    # print(f"原版本时间: {time1:.4f}s")
    # print(f"优化版本时间: {time2:.4f}s")
    # print(f"加速比: {time1 / time2:.2f}x")
    # print(f"结果一致: {np.allclose(patches1, patches2)}")
    # --------------------------------
