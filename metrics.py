import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
'''
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
def recall_coef(output, target):
    output = (torch.sigmoid(output) >= 0.5).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例和假反例
    true_positives = (output * target).sum()  # 正确预测的正例数
    false_negatives = ((1 - output) * target).sum()  # 没有预测出来的正例数

    # 避免除以零的情况
    smooth = 1e-5
    recall_score = true_positives / (true_positives + false_negatives + smooth)

    return recall_score
'''
'''
import torch
import torch.nn.functional as F

def iou_score(output, target, smooth=1.0):
    """
    计算二分类的 IoU
    output: 模型输出的预测结果，形状为 [B, 1, H, W]
    target: 真实标签，形状为 [B, 1, H, W]
    """
    target = target.squeeze(1).float()
    pred = torch.sigmoid(output)
    pred = (pred >= 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()

def dice_coef(output, target, smooth=1.0):
    """
    计算二分类的 Dice 系数
    output: 模型输出的预测结果，形状为 [B, 1, H, W]
    target: 真实标签，形状为 [B, 1, H, W]
    """
    target = target.squeeze(1).float()
    pred = torch.sigmoid(output)
    pred = (pred >= 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


'''
def dice_coef(pred, target):
        smooth = 1.0
        target = target.squeeze(1).float()
        pred = torch.sigmoid(pred)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union +smooth)
        return dice.item()

def iou_score(pred, target):
    smooth=1.0
    target = target.squeeze(1).float()
    pred = torch.sigmoid(pred)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    den = union - intersection
    iou = (intersection + smooth) / (den + smooth)
    return iou.item()

def recall_coef(output, target):
    output = (torch.sigmoid(output) >= 0.5).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例和假反例
    true_positives = (output * target).sum()  # 正确预测的正例数
    false_negatives = ((1 - output) * target).sum()  # 没有预测出来的正例数

    # 避免除以零的情况
    smooth = 1e-5
    recall_score = true_positives / (true_positives + false_negatives + smooth)

    return recall_score
def accuracy(pred, target):
    pred = (torch.sigmoid(pred) >= 0.5).view(-1)  # Sigmoid后转化为二值，视作二分类
    target = target.view(-1)
    correct = (pred == target).sum().item()  # 预测正确的像素数
    total = target.numel()  # 总的像素数
    acc = correct / total  # 准确率
    return acc

