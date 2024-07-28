import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.FI_models.EDSCVgg import Model as EDSC
from model.FI_models.IFRNetVgg import Model as IFRNet
from model.FI_models.RIFEVgg import Model as RIFE
from model.FI_models.AMTVgg import Model as AMT
from model.FI_models.UPRNetVgg import Model as UPRNet
from model.FI_models.EMAVFIVgg import Model as EMAVFI

from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import DZDataset

device = torch.device("cuda")


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, data_root, log_dir, local_rank):
    step = 0
    dataset = DZDataset('train', data_root=data_root)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data

            b, t, c, h, w = data_gpu.shape
            data_gpu = data_gpu.view(-1, c, h, w)
            timestep = timestep.view(-1, timestep.shape[-3], timestep.shape[-2], timestep.shape[-1])

            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, info = model.update(imgs, gt, timestep=timestep, learning_rate=learning_rate, training=True) # pass timestep if you are training RIFEm
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e} loss_vgg:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1'],  info['loss_vgg']))
            step += 1
        model.save_model(log_dir, local_rank, suffix=str(epoch)) 
        dist.barrier()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="finetune FT model")
    parser.add_argument("--model", type=str, default="RIFE", help="FI model (EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI)")
    parser.add_argument("--log_dir", type=str, default="./ckpt/RIFE_finetuned", help="log path")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/DCSZ_dataset/DCSZ_syn", help="train data path")
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)

    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model == "EDSC":
        model = EDSC(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/EDSC/", rank=args.local_rank)
    elif args.model == "IFRNet":
        model = IFRNet(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/IFRNet/", rank=args.local_rank)
    elif args.model == "RIFE":
        model = RIFE(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/RIFE/", rank=args.local_rank)
    elif args.model == "AMT":
        model = AMT(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/AMT/", rank=args.local_rank)
    elif args.model == "UPRNet":
        model = UPRNet(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/UPRNet/", rank=args.local_rank)
    elif args.model == "EMAVFI":
        model = EMAVFI(args.local_rank)
        model.load_pretrained_model("./pretrained_dirs/EMAVFI/", rank=args.local_rank)
    else:
        print("Warning: unsupported FI model.")
        exit(2)

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    data_root = args.dataset_dir
    train(model, data_root, log_dir, args.local_rank)
        


