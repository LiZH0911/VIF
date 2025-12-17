import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # 训练初始化
    def init_train(self):
        pass

    # 加载模型
    def load(self):
        pass

    # 保存模型
    def save(self, label):
        pass

    # 定义损失函数
    def define_loss(self):
        pass

    # 定义优化器
    def define_optimizer(self):
        pass

    # 定义学习率调度器
    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # 数据预处理
    def feed_data(self, data):
        pass

    # 参数优化
    def optimize_parameters(self):
        pass

    # 获取可视化结果
    def current_visuals(self):
        pass

    # 获取当前损失
    def current_losses(self):
        pass

    # 更新学习率
    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    # 当前学习率
    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    # 梯度控制
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """
    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    # 去除 DataParallel 或 DistributedDataParallel 包装，获取实际模型对象
    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        if self.opt['dist']:
            # 分布式训练
            find_unused_parameters = self.opt['find_unused_parameters']
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        else:
            # 单机多卡训练
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n' # 网络类名
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n' # 参数总量
        msg += 'Net structure:\n{}'.format(str(network)) + '\n' # 网络结构
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        # 创建表头（使用格式化字符串）
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        # 遍历模型的所有参数
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float() # 克隆参数数据并转换为float类型
                # 格式化输出参数统计信息
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg


    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu() # 将所有参数移动到CPU
        torch.save(state_dict, save_path)


    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict: # 严格模式：要求完全匹配
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else: # 非严格模式：允许部分匹配
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    # def update_E(self, decay=0.999):
    #     netG = self.get_bare_model(self.netG)
    #     netG_params = dict(netG.named_parameters())
    #     netE_params = dict(self.netE.named_parameters())
    #     for k in netG_params.keys():
    #         netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)





# # 定义一个简单的网络
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc = nn.Linear(32 * 8 * 8, 10)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)
#
# def get_bare_model(network):
#     """Get bare model, especially under wrapping with
#     DistributedDataParallel or DataParallel.
#     """
#     if isinstance(network, (DataParallel, DistributedDataParallel)):
#         network = network.module
#     return network
#
# def describe_network(network):
#     network = get_bare_model(network)
#     msg = '\n'
#     msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n' # 网络类名
#     msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
#     msg += 'Net structure:\n{}'.format(str(network)) + '\n'
#     return msg
#
# def describe_params(network):
#     network = get_bare_model(network)
#     msg = '\n'
#     msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
#     for name, param in network.state_dict().items():
#         if not 'num_batches_tracked' in name:
#             v = param.data.clone().float()
#             msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
#     return msg
#
# def save_network(save_dir, network, network_label, iter_label):
#     save_filename = '{}_{}.pth'.format(iter_label, network_label)
#     save_path = os.path.join(save_dir, save_filename)
#     network = get_bare_model(network)
#     state_dict = network.state_dict()
#     for key, param in state_dict.items():
#         state_dict[key] = param.cpu() # 将所有参数移动到CPU
#     torch.save(state_dict, save_path)
#
# if __name__ == '__main__':
#     # 创建模型并使用DataParallel包装
#     model = SimpleCNN()
#     model_parallel = DataParallel(model, device_ids=[0])
#
#     # 使用describe_network函数
#     info = describe_params(model_parallel)
#     print(info)
#
#     # print(model.state_dict())
#
#     save_network(save_dir='./', network=model, network_label='test_net', iter_label='epoch_0')

