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

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    trainpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/train_10.txt"
    valpath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/txtlabel/val.txt"
    savepath = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/LCD_10.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using "+str(device))
    net=LCDNet()
    net.to(device)

    train_dataset = MyDataset(trainpath,transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = MyDataset(valpath,transforms=get_transform(train=False, mean=mean, std=std))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=4, shuffle=True)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_steps = len(train_loader)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    epochs = 50
    best_acc = 0.0
    seg_lossfunc = nn.BCELoss()
    cla_lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, masks, labels = data
            optimizer.zero_grad()
            cla_out, seg_out = net(images.to(device))
            cla_loss = cla_lossfunc(cla_out,labels.to(device))
            # seg_loss = seg_lossfunc(seg_out,masks.to(device))
            seg_loss = nn.functional.cross_entropy(seg_out, masks.to(device), ignore_index=255, weight=None)
            loss = (cla_loss + seg_loss) / 2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} cla_loss:{:.3f} seg_loss:{:.3f}".format(epoch + 1,epochs,loss,cla_loss,seg_loss)
        
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

if __name__ == "__main__":
    main()