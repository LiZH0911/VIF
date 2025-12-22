import os
import torch
import argparse
from torch.utils.data import DataLoader
import datetime
from utils import utils_image
from torch.optim import Adam, lr_scheduler
from network.net_lzh-fusion import lzh-fusion

from utils.dataset_vif import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # args
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', type=str, default='lzh_fusion')
    parser.add_argument('--save_model_dir', type=str,default='./models/')
    parser.add_argument('--height', type=int, default=256, help='训练图像的高')
    parser.add_argument('--width', type=int, default=256, help='训练图像的宽')
    parser.add_argument('--in_channel', type=int, default=1,
                        help='3 means color image and 1 means gray image')

    # device
    parser.add_argument('--device_ids', type=int, default=[0], help='GPU id list')

    # dataset
    parser.add_argument('--dataset_name', type=str, default='MSRS')
    # parser.add_argument('--dataset_root_dir', type=str, default='E:/coco/train2014/train2014', help='训练图像的目录')
    parser.add_argument('--dataset_root_dir', type=str, default='Dataset/train/MSRS/')
    parser.add_argument('--train_num', type=int, default=40000, help='训练图像的数量')
    parser.add_argument('--type_A', type=str, default='ir')
    parser.add_argument('--type_B', type=str, default='vi')
    parser.add_argument('--dir_A', type=str, default='Dataset/train/MSRS/ir/')
    parser.add_argument('--dir_B', type=str, default='Dataset/train/MSRS/vi/')

    # optimizer
    parser.add_argument('--lr_start', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='学习率衰减因子')

    # train
    parser.add_argument('--num_epochs', type=int, default=4, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('--checkpoint_print', type=int, default=200, help='打印训练过程信息的batch间隔')
    parser.add_argument('--checkpoint_save', type=int, default=1, help='保存模型的epoch间隔')

    args = parser.parse_args()


    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = lzh_fusion().to(device)
    # model = nn.DataParallel(model, device_ids=args.device_ids)


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
    train_dataset = Dataset(dir_A=args.dir_A, dir_B=args.dir_B, in_channel=args.in_channel, pair_num=args.train_num)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    num_batches = len(train_dataloader)

    # Record the accumulated loss within a step length to calculate the printed average loss
    total_loss_accumulate = 0.

    # save
    utils_image.FileHandler.make_dir(args.save_model_dir)

    timestamp = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
    current_step = 0
    for epoch_idx in range(args.num_epochs): # 遍历每个轮次
        print("-------epoch {} start-------\n".format(epoch_idx + 1))
        model.train()
        lr_this_epo = optimizer.param_groups[0]['lr']
        for batch_idx, (batch_A, batch_B)  in enumerate(train_dataloader):
            # current batch
            batch_A.to(device)
            batch_B.to(device)
            # forward
            outputs = model(batch_A, batch_B)
            # current loss
            total_loss = outputs['total_loss']
            # update optimizer
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # accumulated loss
            total_loss_accumulate += total_loss
            current_step += 1
            # print training information
            if current_step % args.checkpoint_print == 0:
                # average loss
                total_loss_avg = total_loss_accumulate / args.checkpoint_print
                msg = "epoch: {}/{}, batch: {}/{}, lr:{:.6f}, total loss: {:.6f} \n". \
                    format(epoch_idx+1, args.num_epochs, batch_idx+1, num_batches, lr_this_epo, total_loss_avg)
                print(msg)

        # adjust the learning rate
        scheduler.step()

        # save model
        if (epoch_idx+1) % args.checkpoint_save == 0:
            model_filename = args.model_name + "_epoch" + str(epoch_idx + 1) + timestamp + ".pth"  # 保存为.pth格式模型
            model_path = os.path.join(args.save_model_dir, model_filename)
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":

    main()