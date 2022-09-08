import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm
from model.MyDataset import MyDataset
# from model.mymodel import LCDNet
from model.mymodelv2 import LCDNet
import transforms as T
# from model.my_dataset import DriveDataset

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

def dice_sim(pred, truth):
    epsilon = 1e-8
    num_batches = pred.size(0)
    m1 = pred.view(num_batches, -1).bool()
    m2 = truth.view(num_batches, -1).bool()

    intersection = torch.logical_and(m1, m2).sum(dim=1)
    return (((2. * intersection + epsilon) / (m1.sum(dim=1) + m2.sum(dim=1) + epsilon)).sum(dim=0))/2

def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)

def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size

def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]

def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
<<<<<<< HEAD

    batch_size = 4

    trainpath = "/home/sda/Users/YT/LCDnet/txtlabel/train_10.txt"
    valpath = "/home/sda/Users/YT/LCDnet/txtlabel/train_10.txt"
    savepath = "/home/sda/Users/YT/LCDnet/weight/LCD_only_seg.pth"

    # trainpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/train_10.txt"
    # valpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/va.txt"
    # savepath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/LCD_10.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
=======
    # trainpath = "/home/sda/Users/YT/LCDnet/train.txt"
    # valpath = "/home/sda/Users/YT/LCDnet/val.txt"
    # savepath = "/home/sda/Users/YT/LCDnet/LCD_11.pth"
    trainpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/train_10.txt"
    valpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/val.txt"
    savepath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/LCD_10.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>>>>>> 21a6017ebe93ba7e97222f4aac9cfd5e208d4bc2
    print("using "+str(device))
    net=LCDNet()
    net.to(device)

    train_dataset = MyDataset(trainpath,transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = MyDataset(valpath,transforms=get_transform(train=False, mean=mean, std=std))
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=6, shuffle=True)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_steps = len(train_loader)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    epochs = 100
    best_acc = 0.0
    cla_lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    for epoch in range(epochs):

        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, masks, labels = data
            optimizer.zero_grad()
            cla_out, seg_out = net(images.to(device))
            balance_para = 0.7
            # cla_loss
            cla_loss = cla_lossfunc(cla_out,labels.to(device))

            # seg_loss 
            loss_weight = torch.as_tensor([1.0, 2.0], device=device)
            dice_target = build_target(masks.to(device), 2, 255)
            seg_loss = nn.functional.cross_entropy(seg_out, masks.to(device), ignore_index=255, weight=loss_weight)
            seg_loss += dice_loss(seg_out, dice_target, multiclass=True, ignore_index=255)

            # loss = cla_loss + balance_para*seg_loss
            loss = seg_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} cla_loss:{:.3f} seg_loss:{:.3f}".format(epoch + 1,epochs,loss,cla_loss,seg_loss)
        scheduler.step()

        # validate
        net.eval()
        acc = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_masks, val_labels = val_data
                cla_out, seg_out = net(val_images.to(device))
                pred_cla = torch.max(cla_out,dim=1)[1]
                acc += torch.eq(pred_cla, val_labels.to(device)).sum().item()
                # 计算dice

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), savepath)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
