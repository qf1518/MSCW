import os
import argparse
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as albu
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
import archs
import losses
from dataset import CustomDataset
from metrics import iou_score, dice_coef, recall_coef, accuracy
from utils import AverageMeter, str2bool
import yaml
import pandas as pd
from collections import OrderedDict
from ducknet import DuckNet

from archs import NestedUNet
from modeling.deeplab import DeepLab
from MSCWNet import MSCWNet
class Config:
    @staticmethod
    def from_cmdline():
        parser = argparse.ArgumentParser(description='Training configuration')
        parser.add_argument('--name', default=None, help='Model name: (default: arch+timestamp)')
        parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs to run')
        parser.add_argument('-b', '--batch_size', type=int, default=4
                            , help='Mini-batch size (default: 16)')
        parser.add_argument('--arch', default='UNet', choices=['UNet', 'NestedUNet','NestedUNetmanba','DuckNet','Unet_scSE_hyper','Unet_1','DeepLab','Unet_0','Unet32'], help='Model architecture')
        parser.add_argument('--deep_supervision', type=str2bool, default=False, help='Use deep supervision if True')
        parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
        parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
        parser.add_argument('--input_w', type=int, default=256, help='Input image width')
        parser.add_argument('--input_h', type=int, default=256, help='Input image height')
        parser.add_argument('--loss', default='BCEDiceLoss', choices=['BCEDiceLoss','LovaszHingeLoss','FocalLoss2d','WeightedBCELoss2d'], help='Loss function')
        parser.add_argument('--dataset', default='Glas_256train', help='CVC-ClinicDB_256 Glas_256train kv_256')
        parser.add_argument('--img_ext', default='.png', help='Image file extension')
        parser.add_argument('--mask_ext', default='.png', help='Mask file extension')
        parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'], help='Optimizer type')
        parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay rate')
        parser.add_argument('--nesterov', type=str2bool, default=False, help='Nesterov momentum')
        parser.add_argument('--scheduler', default='CosineAnnealingLR',
                            choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'],
                            help='Learning rate scheduler')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
        parser.add_argument('--factor', type=float, default=0.1, help='Factor for ReduceLROnPlateau')
        parser.add_argument('--patience', type=int, default=2, help='Patience for ReduceLROnPlateau')
        parser.add_argument('--milestones', type=str, default='1,2', help='Milestones for MultiStepLR')
        parser.add_argument('--gamma', type=float, default=2 / 3, help='Gamma for MultiStepLR')
        parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping threshold')
        parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')

        args = parser.parse_args()
        return vars(args)


class ModelManager:
    def __init__(self, config):
        self.config = config
        if config['arch'] == 'NestedUNet':
            self.model = NestedUNet(num_classes=1).cuda()
        elif config['arch'] == 'MSCWNet':
            #self.model = UNet(num_classes=1)
            self.model = MSCWNet().cuda()
        elif config['arch'] == 'Unet32':
            self.model = Unet32().cuda()
        elif config['arch'] == 'UnetDCA':
            self.model = UnetDCA().cuda()
        elif config['arch'] == 'Unet_0':
            self.model = Unet_0().cuda()
        elif config['arch'] == 'NestedUNetmanba':
            self.model = NestedUNetmanba(num_classes=1).cuda()
        elif config['arch'] == 'DuckNet':
            self.model = DuckNet().cuda()
        elif config['arch'] == 'SwinUnet':
            self.model = SwinUnet().cuda()
        elif config['arch'] == 'TransUnet':
            self.model = TransUnet().cuda()
        elif config['arch'] == 'DeepLab':
            self.model = DeepLab(num_classes=1, backbone='resnet', output_stride=16, sync_bn=None,
                        freeze_bn=False).cuda()

        self.criterion = self.create_criterion().cuda()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()

    def create_criterion(self):
        if self.config['loss'] == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss()
        else:
            return losses.__dict__[self.config['loss']]()

    def create_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.config['optimizer'] == 'Adam':
            return optim.Adam(params, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            return optim.SGD(params, lr=self.config['lr'], momentum=self.config['momentum'],
                             nesterov=self.config['nesterov'], weight_decay=self.config['weight_decay'])

    def create_scheduler(self):
        if self.config['scheduler'] == 'CosineAnnealingLR':
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs'], eta_min=self.config['min_lr'])
        elif self.config['scheduler'] == 'ReduceLROnPlateau':
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=self.config['factor'], patience=self.config['patience'],
                min_lr=self.config['min_lr'])
        elif self.config['scheduler'] == 'MultiStepLR':
            milestones = list(map(int, self.config['milestones'].split(',')))
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=self.config['gamma'])


class DataManager:
    def __init__(self, config):
        self.config = config
        self.train_loader, self.val_loader = self.setup_loaders()

    def setup_loaders(self):

        img_ids = glob(os.path.join('inputs', self.config['dataset'], 'images', '*' + self.config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        train_transform = Compose([
            albu.RandomRotate90(), albu.Flip(),
            OneOf([albu.HueSaturationValue(), albu.RandomBrightnessContrast()], p=1),
            albu.Resize(self.config['input_h'], self.config['input_w']), albu.Normalize()])

        val_transform = Compose([
            albu.Resize(self.config['input_h'], self.config['input_w']), albu.Normalize()])

        train_dataset = CustomDataset(img_ids=train_img_ids,
                                      img_dir=os.path.join('inputs', self.config['dataset'], 'images'),
                                      mask_dir=os.path.join('inputs', self.config['dataset'], 'masks'),
                                      img_ext=self.config['img_ext'],
                                      mask_ext=self.config['mask_ext'],
                                      num_classes=self.config['num_classes'],
                                      transform=train_transform)
        val_dataset = CustomDataset(img_ids=val_img_ids,
                                    img_dir=os.path.join('inputs', self.config['dataset'], 'images'),
                                    mask_dir=os.path.join('inputs', self.config['dataset'], 'masks'),
                                    img_ext=self.config['img_ext'],
                                    mask_ext=self.config['mask_ext'],
                                    num_classes=self.config['num_classes'],
                                    transform=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                                   shuffle=True, num_workers=self.config['num_workers'], drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                                 shuffle=False, num_workers=self.config['num_workers'], drop_last=False)
        return train_loader, val_loader


def main():
    config = Config.from_cmdline()
    manager = ModelManager(config)
    data_manager = DataManager(config)
    file_name = config['arch'] +'_base'
    print(file_name)
    os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)
    # save configuration
    with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_recall'])
    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        train_log = train_epoch(data_manager.train_loader, manager.model, manager.criterion,
                                            manager.optimizer, config)
        val_log = validate_epoch(data_manager.val_loader, manager.model, manager.criterion, config)
        print(
            'Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f},Validation Recall:{:.4f}'.format(
                epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'],
                val_log['dice'], val_log['iou'], val_log['recall']))

        '''
        # Update scheduler, save models, etc.
        if config['scheduler'] == 'ReduceLROnPlateau':
            manager.scheduler.step(val_loss)
        else:
            manager.scheduler.step()
            '''
        tmp = pd.Series([
            epoch,
            config['lr'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
            val_log['recall']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice', 'val_recall'])
        log = log._append(tmp, ignore_index=True)
        log.to_csv('model_outputs/{}/log1.csv'.format(file_name), index=False)
        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(manager.model.state_dict(), 'model_outputs/{}/model.pth'.format(file_name))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()


def train_epoch(train_loader, model, criterion, optimizer, config):
        model.train()
        avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(), 'acc': AverageMeter()}
        pbar = tqdm(total=len(train_loader), desc='Train')

        for data in train_loader:
            if len(data) == 2:
                input, target = data
            elif len(data) > 2:
                input, target, _ = data  # 根据实际返回的数据格式解包
            input = input.cuda()
            target = target.cuda()

            # Compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(output, target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
            acc = accuracy(output, target)

            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))
            pbar.update(1)
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('acc', avg_meters['acc'].avg),  # 添加accuracy
            ])
            pbar.set_postfix(postfix)

        pbar.close()
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('acc', avg_meters['acc'].avg)])  # 返回accuracy

def validate_epoch(val_loader, model, criterion, config):
        model.eval()
        avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(),'dice': AverageMeter(),'recall':AverageMeter(),'acc': AverageMeter()}
        pbar = tqdm(total=len(val_loader), desc='Validate')

        with torch.no_grad():
            for data in val_loader:
                if len(data) == 2:
                    input, target = data
                elif len(data) > 2:
                    input, target, _ = data  # 根据实际返回的数据格式解包
                input = input.cuda()
                target = target.cuda()

                # Compute output
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
                    dice = dice_coef(output, target)
                    recall = recall_coef(output, target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou = iou_score(output, target)
                    dice = dice_coef(output, target)
                    recall = recall_coef(output, target)
                    acc = accuracy(output, target)

                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                avg_meters['dice'].update(dice, input.size(0))
                avg_meters['recall'].update(recall, input.size(0))
                avg_meters['acc'].update(acc, input.size(0))
                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('dice', avg_meters['dice'].avg),
                    ('recall', avg_meters['recall'].avg),
                    ('acc', avg_meters['acc'].avg),  # 添加accuracy
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('recall', avg_meters['recall'].avg),
                            ('acc', avg_meters['acc'].avg)])

    # Training and validation logic can go here using manager and data_manager


if __name__ == '__main__':
    main()
