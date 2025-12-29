import os
import torch
import argparse
import time
from utils.dataset_vif import Dataset
from torch.utils.data import DataLoader
from utils import utils_image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # args
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_path', type=str,default='./models/')
    parser.add_argument('--in_channel', type=int, default=1,
                        help='3 means color image and 1 means gray image')

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
    dir_A = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_A)
    dir_B = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_B)
    test_set = Dataset(dir_A=dir_A, dir_B=dir_B, in_channel=args.in_channel)
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

    with torch.no_grad():
        for i, img_A, img_B in enumerate(test_loader):
            img_A.to(device)
            img_A.to(device)
            start = time.time()

            # inference
            # encoder
            feature_A = model.encoder(img_A)
            feature_B = model.encoder(img_B)

            # fusion
            feature_fused = model.fusion(img_A, img_B, in_channel=args.in_channel)

            # decoder
            img_fusion = model.decoder(feature_fused)

            # save
            output = utils_image.FormatConversion.tensor2uint(output)
            output_path = os.path.join(save_dir, args.dataset_name, img_name + '.png')
            utils_image.FileHandler.save_img(output, output_path)
            end = time.time()
            print(end-start)

if __name__ == '__main__':
    main()