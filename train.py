import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import transforms as Trans


from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model.MyDataset import MyDataset
from model.mymodelv2 import LCDNet
from train_utils.train_and_eval import create_lr_scheduler,evaluate,train_one_epoch 
from train_utils.dice_coefficient_loss import build_target, dice_coeff, multiclass_dice_coeff,dice_loss


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return Trans.SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return Trans.SegmentationPresetEval(mean=mean, std=std)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    args = parser.parse_args()
    return args

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    batch_size = 2
    seg_num_classes = 2
    cla_num_classes = 3
    results_file = "/home/sda/Users/YT/LCDnet/Log/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    trainpath = "/home/sda/Users/YT/LCDnet/txtlabel/train_10.txt"
    valpath = "/home/sda/Users/YT/LCDnet/txtlabel/train_10.txt"
    savepath = "/home/sda/Users/YT/LCDnet/weight/LCD_loc_v7.pth"

    # trainpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/train_10.txt"
    # valpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/va.txt"
    # savepath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/LCD_10.pth"

    # model config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using "+str(device))
    net=LCDNet()
    net.to(device)

    # dataset config
    train_dataset = MyDataset(trainpath,transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = MyDataset(valpath,transforms=get_transform(train=False, mean=mean, std=std))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # classifaction
    best_acc = 0.0

    # segmentation
    best_dice = 0.0

    # train
    epochs = 100
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]

    params_to_optimize_loc  = [p for p in net.fc.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    optimizer2 = torch.optim.SGD(
        params_to_optimize_loc,
        lr=0.01, momentum=0.9, weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler() if False else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 100, warmup=True)
    start_time = time.time()
    for epoch in range(epochs):

        # train
        seg_loss, lr = train_one_epoch(net, optimizer,optimizer2, train_loader, device, epoch, cla_num_classes,seg_num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=100, scaler=scaler)
        cla_confmat,seg_confmat,dice = evaluate(net, val_loader, device=device, cla_num_classes=cla_num_classes,seg_num_classes=seg_num_classes,seg=True,cla=True)
        cla_val_info = str(cla_confmat)
        seg_val_info = str(seg_confmat)
        print(cla_val_info)
        print(seg_val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                        f"train_loss: {seg_loss:.4f}\n" \
                            f"lr: {lr:.6f}\n" \
                            f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + cla_val_info + seg_val_info+"\n\n")

        torch.save(net.state_dict(), savepath)
        

if __name__ == "__main__":
    main()
