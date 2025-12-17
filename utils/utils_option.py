import os
from datetime import datetime
import json
import re
import glob

def get_timestamp():
    return datetime.now().strftime('_%Y%m%d_%H%M%S')

# 通过json文件构造一个字典对象opt，记录模型配置信息
def parse(json_path, is_train=True):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(json_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n' # 去除每行后面的注释，获得标准JSON文件
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str)

    opt['json_path'] = json_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    if 'dist' not in opt:
        opt['dist'] = False

    opt['num_gpu'] = len(opt['gpu_ids'])
    print('number of GPUs is: ' + str(opt['num_gpu']))


    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0] # train or test
        dataset['phase'] = phase
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if 'dir_A' in dataset and dataset['dir_A'] is not None:
            dataset['dir_A'] = os.path.expanduser(dataset['dir_A'])
        if 'dir_B' in dataset and dataset['dir_B'] is not None:
            dataset['dir_B'] = os.path.expanduser(dataset['dir_B'])

    # ----------------------------------------
    # path
    # ----------------------------------------

    return opt

'''
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
'''
def save(opt, save_dir):
    json_path = opt['json_path']
    dirname, filename_ext = os.path.split(json_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(save_dir, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)

