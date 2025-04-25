import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','FocalLoss2d']

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = torch.softmax(logit,1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob*select).sum(1).view(-1,1)
        prob = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class WeightedBCELoss2d(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        自定义加权二元交叉熵损失函数，适用于二维图像分割任务。

        Args:
            alpha (list or None): 类别权重列表，长度应为2（背景和前景）。例如 [0.3, 0.7]。
                                   如果为 None，则不应用类别权重。
            gamma (float): Focal Loss 的 gamma 参数，用于调节难易样本的权重。
                           默认值为2。
            reduction (str): 聚合方式，选择 'mean'、'sum' 或 'none'。
        """
        super(WeightedBCELoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None:
            assert isinstance(self.alpha, (list, tuple)), "alpha 必须是列表或元组"
            assert len(self.alpha) == 2, "对于二分类，alpha 长度必须为2"
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, input, target):
        """
        计算加权的二元交叉熵损失。

        Args:
            input (Tensor): 模型输出 logits，形状为 (N, 1, H, W)。
            target (Tensor): 真实标签，形状为 (N, 1, H, W)，取值为0或1。

        Returns:
            Tensor: 计算得到的损失值。
        """
        device = input.device
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            alpha_t = alpha[target.long()]
        else:
            alpha_t = torch.ones_like(input, device=device)

        # 使用 BCEWithLogitsLoss 计算基础的二元交叉熵损失
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(input, target)

        # 计算预测概率
        probs = torch.sigmoid(input)
        probs = probs.view(-1)
        target = target.view(-1).float()

        # 计算 pt
        pt = torch.where(target == 1, probs, 1 - probs)

        # 计算 Focal Loss 权重
        focal_weight = alpha_t.view(-1) * (1 - pt) ** self.gamma

        # 计算加权损失
        loss = focal_weight * bce_loss.view(-1)

        # 应用聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
