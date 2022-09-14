import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.mymodelv2 import LCDNet


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      img_mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # (512,512,3)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    # 单通道变三通道
    img_mask = img_mask[:, :, np.newaxis]
    img_mask = img_mask.repeat([3], axis=2)
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
            
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
    # load image
    img_dir = "/home/sda/Users/YT/DRIVE/training/images_10"
    images = os.listdir(img_dir)
    count0 = 0
    count1 = 0
    count2 = 0 
    count = 0
    for image in images:
        img_path = "/home/sda/Users/YT/DRIVE/training/images_10/verse181_250_training.png"
        # img_path = os.path.join(img_dir,image)
        img_mask_path = "/home/sda/Users/YT/DRIVE/training/label_10/verse001_250_manual1.png"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img_mask = Image.open(img_mask_path).convert('L')
        img_mask = np.array(img_mask) / 255
        # plt.imshow(img)

        img_np = np.array(img)
        img_np = np.float32(img_np)/255
        
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # create model
        model = LCDNet()
        model.to(device)

        # load model weights
        weights_path = "/home/sda/Users/YT/LCDnet/weight/LCD_bp_2_weight_112.pth"
        # weights_path = "/home/sda/Users/YT/LCDnet/weight/LCD_loc_weight111_v5.pth"
        # weights_path = "/home/sda/Users/YT/LCDnet/weight/LCD_loc_test_v4.pth"
        weights_path = "/home/sda/Users/YT/LCDnet/weight/LCD_loc_weight111_v6.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        target_layers = [model.conv5[-1]]
        input_tensor = img
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = None
        # targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # (1,512,512)
        grayscale_cam = grayscale_cam[0, :]
        print(grayscale_cam.shape)
        # plt.imsave("/home/server/Desktop/zky-sxr/yinteng/1.png",grayscale_cam)
        visualization = show_cam_on_image(img_np, grayscale_cam,img_mask, use_rgb=False)
        # (512,512,3)
        cv2.imwrite("/home/sda/Users/YT/LCDnet/predictImg/"+image+"8.png", visualization)

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))[0]).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(str(predict_cla),
                                                    predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(str(i),
                                                    predict[i].numpy()))
        if predict_cla == 0:
            count0 = count0 + 1
        if predict_cla == 1:
            count1 = count1 + 1
        if predict_cla == 2:
            count2 = count2 + 1
        break

if __name__ == '__main__':
    main()
