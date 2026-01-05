import os
import numpy as np
from tqdm import tqdm
import argparse

import utils.utils_image as utils

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def main():
    # args
    parser = argparse.ArgumentParser()

    # original dataset
    parser.add_argument('--dataset_name', type=str, default='MSRS')
    parser.add_argument('--dataset_dir', type=str, default='./MSRS_train/')
    parser.add_argument('--type_A', type=str, default='ir')
    parser.add_argument('--type_B', type=str, default='vi')

    # save
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--output_root_dir', type=str, default='./dataset/train')

    args = parser.parse_args()

    # dataset
    dir_A = os.path.join(args.dataset_dir, args.type_A)
    dir_B = os.path.join(args.dataset_dir, args.type_B)
    # save
    output_dir_A = os.path.join(args.output_root_dir, args.dataset_name+'_imgsize_'+f'{args.img_size}', args.type_A)
    output_dir_B = os.path.join(args.output_root_dir, args.dataset_name+'_imgsize_'+f'{args.img_size}', args.type_B)
    utils.FileHandler.make_dir(output_dir_A)
    utils.FileHandler.make_dir(output_dir_B)

    paths_A, names_A = utils.FileHandler.list_img_paths(dir_A)
    paths_B, names_B = utils.FileHandler.list_img_paths(dir_B)

    for i in tqdm(range(len(paths_A))):
        # read image
        img_A = utils.FileHandler.imread_uint(paths_A[i], n_channels=1) # ir uint8 G
        img_B = utils.FileHandler.imread_uint(paths_B[i], n_channels=3) # vi uint8 RGB
        # RGB to Y
        img_B = utils.ImageChannelConversion.rgb2ycbcr(img_B, only_y=True) # vi uint8 Y
        img_B = np.expand_dims(img_B, axis=2)

        # crop
        img_A_Patch_Group, _ = utils.ImageProcess.img_patching(img_A, c_h=args.img_size, c_w=args.img_size)
        img_B_Patch_Group, _ = utils.ImageProcess.img_patching(img_B, c_h=args.img_size, c_w=args.img_size)

        print(img_A_Patch_Group.shape)

        for j in range(img_A_Patch_Group.shape[0]):
            bad_A = is_low_contrast(img_A_Patch_Group[j, :, :, :])
            bad_B = is_low_contrast(img_B_Patch_Group[j, :, :, :])
            # print(not (bad_A or bad_B))
            # Determine if the contrast is low
            if not (bad_A or bad_B):
                avl_A_Patch = img_A_Patch_Group[j, :, :, :]
                avl_B_Patch = img_B_Patch_Group[j, :, :, :]
                # print(avl_A_Patch.shape)
                output_path_A = os.path.join(output_dir_A, names_A[i] + f'_{j}' + '.png')
                output_path_B = os.path.join(output_dir_B, names_B[i] + f'_{j}' + '.png')
                utils.FileHandler.save_img(avl_A_Patch, output_path_A)
                utils.FileHandler.save_img(avl_B_Patch, output_path_B)


if __name__ == "__main__":

    # # 模拟一个低对比度图像（像素值集中在120-130之间）
    # image = np.random.randint(120, 131, size=(100, 100, 1))
    # print(is_low_contrast(image))  # 可能返回 True
    #
    # # 模拟一个高对比度图像（像素值范围广）
    # image = np.random.randint(0, 256, size=(100, 100))
    # print(is_low_contrast(image))  # 可能返回 False

    main()