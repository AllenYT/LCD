import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.mymodelv2 import LCDNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
    transforms.ToTensor()
])
    # load image
    img_path = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/dest/images/verse005_270_training.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
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

    target_layers = [model.conv5[-1]]
    input_tensor = img
    input_tensor = input_tensor
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    # plt.imsave("/home/server/Desktop/zky-sxr/yinteng/1.png",grayscale_cam)
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=False)
    cv2.imwrite("/home/server/Desktop/zky-sxr/yinteng/01.png", visualization)

    # load model weights
    weights_path = "/home/server/Desktop/zky-sxr/yinteng/LCDNet/LCD_10.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

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


if __name__ == '__main__':
    main()
