import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import datetime
from utils import utils_image
from torch.optim import Adam, lr_scheduler
from network.loss import Fusionloss
from network.net_lzh-fusion import lzh-fusion
from tensorboardX import SummaryWriter
import logging
from logger import setup_logging
from utils.fix_random_seed import set_seed

from utils.dataset_vif import Fusion_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main():
    # args
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', type=str, default='lzh_fusion')
    parser.add_argument('--save_model_dir', type=str,default='./models/')
    # parser.add_argument('--height', type=int, default=256, help='训练图像的高')
    # parser.add_argument('--width', type=int, default=256, help='训练图像的宽')
    # parser.add_argument('--in_channels', type=int, default=1,
    #                     help='3 means color image and 1 means gray image')

    # device
    parser.add_argument('--device_ids', type=int, default=[0], help='GPU id list')

    # dataset
    # parser.add_argument('--dataset_root_dir', type=str, default='E:/coco/train2014/train2014', help='训练图像的目录')
    parser.add_argument('--dataset_root_dir', type=str, default='./dataset/train/', help='训练数据集存储根目录')
    parser.add_argument('--dataset_name', type=str, default='MSRS_imgsize_128', help='数据集名称')
    parser.add_argument('--type_A', type=str, default='ir', help='A类图像类型')
    parser.add_argument('--type_B', type=str, default='vi', help='B类图像类型')

    # optimizer
    parser.add_argument('--lr_start', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='学习率衰减因子')

    # train
    parser.add_argument('--seed', type=int, default=3407, help='random seed for training')
    parser.add_argument('--num_epochs', type=int, default=4, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('--checkpoint_print', type=int, default=200, help='打印训练过程信息的batch间隔')
    parser.add_argument('--checkpoint_save', type=int, default=1, help='保存模型的epoch间隔')
    parser.add_argument('--tensorboard_logdir', type=str, default='./train_log/test_log_dir', help='used for tensorboard writer init')
    args = parser.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # seed
    set_seed(args.seed)

    # model
    model = lzh_fusion().to(device)
    # model = nn.DataParallel(model, device_ids=args.device_ids)

    # save
    utils_image.FileHandler.make_dir(args.save_model_dir)

    # logger
    setup_logging(log_dir=args.save_model_dir) # logging config
    logger = logging.getLogger()
    # tensorboard
    writer = SummaryWriter(log_dir=args.tensorboard_logdir)
    timestamp = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S') # timestamp

    # optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr_start, # learning rate
        betas=(0.9, 0.999),  # 动量衰减率
        eps=1e-8, # 数值稳定常数
        weight_decay=1e-4, # L2正则化，防止过拟合
        amsgrad = False  # 不使用AMSGrad变体
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_decay ** epoch)


    # dataset
    dir_ir = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_A)
    dir_vi = os.path.join(args.dataset_root_dir, args.dataset_name, args.type_B)
    train_dataset = Fusion_Dataset(dir_ir=dir_ir, dir_vi=dir_vi)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    num_batches = len(train_dataloader)

    # fusion loss
    criteria_fusion = Fusionloss() # 融合损失使用自定义的Fusionloss
    loss_total_accumulate = 0.  # Record the accumulated loss within a step length to calculate the printed average loss.

    # train
    current_step = 0
    logger.info('Training Fusion Model start~')
    for epoch_idx in range(args.num_epochs):
        logger.info("-------epoch {} start-------\n".format(epoch_idx + 1))
        model.train()
        lr_this_epo = optimizer.param_groups[0]['lr']
        logger.info("learning rate: ",lr_this_epo)

        # add training info to tensorboard writer
        writer.add_scalar(tag='learning_rate', scalar_value=lr_this_epo, global_step=epoch_idx + 1)

        for batch_idx, (imgs_ir, imgs_vi, name)  in enumerate(train_dataloader):
            # current batch
            imgs_ir = imgs_ir.to(device)
            imgs_vi = imgs_vi.to(device)
            imgs_vi_ycrcb = utils_image.ImageChannelConversion.RGB2YCrCb(imgs_vi)

            # forward
            outputs = model(imgs_ir, imgs_vi_ycrcb)

            imgs_fusion_ycrcb = torch.cat(
                (outputs, imgs_vi_ycrcb[:, 1:2, :, :],
                 imgs_vi_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            imgs_fusion = utils_image.ImageChannelConversion.YCrCb2RGB(imgs_fusion_ycrcb)
            ones = torch.ones_like(imgs_fusion)
            zeros = torch.zeros_like(imgs_fusion)
            imgs_fusion = torch.where(imgs_fusion > ones, ones, imgs_fusion) # 将大于1的值设置为1
            imgs_fusion = torch.where(imgs_fusion < zeros, zeros, imgs_fusion) # 将小于0的值设置为0

            # current fusion loss
            loss_total, loss_gradient, loss_l1, loss_SSIM = criteria_fusion(imgs_ir, imgs_vi_ycrcb, outputs)

            # update optimizer
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # accumulated loss
            loss_total_accumulate += loss_total
            current_step += 1

            writer.add_scalar(tag="loss_total", scalar_value=loss_total, global_step=current_step)

            # print training information
            if current_step % args.checkpoint_print == 0:
                # average loss
                total_loss_avg = loss_total_accumulate / args.checkpoint_print

                writer.add_scalar(tag="total_loss_avg", scalar_value=total_loss_avg, global_step=current_step)

                msg = "epoch: {}/{}, batch: {}/{}, lr:{:.6f}, total loss: {:.6f} \n". \
                    format(epoch_idx+1, args.num_epochs, batch_idx+1, num_batches, lr_this_epo, total_loss_avg)
                logger.info(msg)

        # adjust the learning rate
        scheduler.step()

        # save model
        if (epoch_idx+1) % args.checkpoint_save == 0:
            model_filename = args.model_name + "_epoch" + str(epoch_idx + 1) + timestamp + ".pth"  # 保存为.pth格式模型
            model_path = os.path.join(args.save_model_dir, model_filename)
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()