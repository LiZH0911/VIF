import os
import torch
import argparse
import time
from utils.dataset_vif import Fusion_Dataset
from torch.utils.data import DataLoader
from network.net_lzh-fusion import lzh-fusion
from utils import utils_image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # args
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_path', type=str,default='./models/')

    parser.add_argument('--dataset_root_dir', type=str, default='./dataset/test/')
    parser.add_argument('--dataset_name', type=str, default='MSRS')
    parser.add_argument('--type_A', type=str, default='ir')
    parser.add_argument('--type_B', type=str, default='vi')

    parser.add_argument('--save_root_dir', type=str, default='./results/')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = lzh_fusion()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    # load test data
    dir_ir = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_A)
    dir_vi = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_B)
    test_set = Fusion_Dataset(dir_ir=dir_ir, dir_vi=dir_vi)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )
    save_dir = os.path.join(args.save_root_dir, args.dataset_name)
    utils_image.FileHandler.make_dir(save_dir)

    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for i, (img_ir, img_vi, name) in enumerate(test_loader):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            img_vi_ycrcb = utils_image.ImageChannelConversion.RGB2YCrCb(img_vi)
            start = time.time()

            # inference

            # # encoder
            # feature_ir = model.encoder(img_ir)
            # feature_vi = model.encoder(img_vi)
            # # fusion
            # feature_fused = model.fusion(img_ir, img_vi, in_channel=args.in_channel)
            # # decoder
            # img_fusion = model.decoder(feature_fused)

            output = model(img_ir, img_vi_ycrcb)
            img_fusion_ycrcb = torch.cat(
                (output, img_vi_ycrcb[:, 1:2, :, :],
                 img_vi_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            img_fusion = utils_image.ImageChannelConversion.YCrCb2RGB(img_fusion_ycrcb)

            # save
            img_fusion = utils_image.FormatConversion.tensor2uint(img_fusion)
            img_name = name + '.png'
            save_path = os.path.join(save_dir, img_name)
            utils_image.FileHandler.save_img(img_fusion, save_path)
            end = time.time()
            print(end-start)
            test_bar.set_description('Fusion {0} Sucessfully!'.format(img_name))

if __name__ == '__main__':
    main()