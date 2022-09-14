import torch
from torch import nn
import numpy as np
import cv2
import train_utils.distributed_utils as utils
from .loc_loss import *
from .dice_coefficient_loss import dice_loss, build_target
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

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
    # img_mask = img_mask[:, :, np.newaxis]
    # img_mask = img_mask.repeat([3], axis=2)
    # if np.max(img) > 1:
    #     raise Exception(
    #         "The input image should np.float32 in the range [0, 1]")
            
    cam = heatmap 
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    loss = 0.0
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)

    return loss


def evaluate(model, data_loader, device, cla_num_classes,seg_num_classes, seg, cla):
    model.eval()
    cla_confmat = utils.ConfusionMatrix(cla_num_classes,seg=False,cla=True)
    seg_confmat = utils.ConfusionMatrix(seg_num_classes,seg=True,cla=False)
    dice = utils.DiceCoefficient(num_classes=seg_num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = ''
    if seg:
        header = 'Train Seg Evaluation:'
        with torch.no_grad():
            for image, target ,label in metric_logger.log_every(data_loader, 100, header):
                image, target ,label = image.to(device), target.to(device),label.to(device)
                output = model(image)[1]

                seg_confmat.update(target.flatten(), output.argmax(1).flatten())
                dice.update(output, target)

            seg_confmat.reduce_from_all_processes()
            dice.reduce_from_all_processes()
    if cla:
        header = 'Train Cla Evaluation:'
        with torch.no_grad():
            for image, target ,label in metric_logger.log_every(data_loader, 100, header):
                image, target ,label = image.to(device), target.to(device),label.to(device)
                output = model(image)[0]

                cla_confmat.update(label.flatten(), output.argmax(1).flatten())

            cla_confmat.reduce_from_all_processes()
    return cla_confmat, seg_confmat, dice.value.item()


def train_one_epoch(model, optimizer, optimizer2,data_loader, device, epoch, cla_num_classes,seg_num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    balance_parameter = 2
    cla_loss_weight = torch.as_tensor([1.0, 1.0, 1.0], device=device)
    
    if seg_num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target,label in metric_logger.log_every(data_loader, print_freq, header):
        image, target, label = image.to(device), target.to(device), label.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            cla_out, seg_out = model(image)
            cla_func = nn.CrossEntropyLoss(weight=cla_loss_weight)
            cla_loss = cla_func(cla_out, label)
            seg_loss = criterion(seg_out, target, loss_weight, num_classes=seg_num_classes, ignore_index=255)
            # Multi Branch Loss
            loss = cla_loss + balance_parameter*seg_loss
            # Location Loss
            target_layers = [model.conv5[-1]]
            input_tensor = image
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # (4,480,480)
            loc_loss_func = Loc_Loss()
            loc_loss = loc_loss_func(grayscale_cam,target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loc_loss.requires_grad_()
            loc_loss.backward()
            optimizer2.step()

        # lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


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
