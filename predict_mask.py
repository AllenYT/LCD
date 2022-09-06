import os
import time
from model.mymodelv2 import LCDNet
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    data_transform = transforms.Compose([transforms.ToTensor()])
    classes = 1 
    weights_path = "/home/sda/Users/YT/LCDnet/LCD_11.pth"
    img_path = "/home/sda/Users/YT/DRIVE/training/imagespine/verse001_250_training.png"
    # roi_path = "/home/sda/Users/YT/unet/DRIVE/test/label/verse042_250_manual1.png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_path), f"image {roi_path} not found."
   
    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = LCDNet()

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device)

    # # load roi mask
    # roi_img = Image.open(roi_path).convert('L')
    # roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        t_start = time_synchronized()
        output = model(img.to(device))[1]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        output = output.argmax(1).squeeze(0)
        # 将前景对应的像素值改成255(白色)
        # print(torch.min(output))
        output = output.to("cpu").numpy().astype(np.uint8)
        output[output == 0 ] = 0
        # # # 将不敢兴趣的区域像素设置成0(黑色)
        output[output == 1 ] = 255
        mask = Image.fromarray(output)
        mask.save("test_result2.png")

if __name__ == '__main__':
    main()