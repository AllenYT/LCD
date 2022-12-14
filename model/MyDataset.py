import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

tot = transforms.Compose([
    transforms.ToTensor()
])

def keep_image_size_open(path,size=(256,256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask

class MyDataset(Dataset):
    def __init__(self,path,transforms):
        self.path = path
        with open(path, "r") as f:
            lines = []
            for line in f.readlines():
                line = line.strip('\n').split(" ")  #去掉列表中每一个元素的换行符
                lines.append(line)
        self.data = lines
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, targets,labels = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_labels = cat_list(labels, fill_value=255)
        return batched_imgs, batched_targets, batched_labels

    def __getitem__(self, index):
        image_path = self.data[index][0]
        mask_path = self.data[index][1]
        frac_class = self.data[index][2]
        image = Image.open(image_path).convert('RGB')
        segment_image = Image.open(mask_path).convert('L')
        segment_image = np.array(segment_image) / 255
        segment_image = Image.fromarray(segment_image)
        img, mask = self.transforms(image,segment_image)
        return img,mask, np.array(int(frac_class))

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == "__main__":
    path = "/home/sda/Users/YT/spine.txt"
