import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from data_utils import get_anchors

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def vgg19_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


class RotatedYoloBase(nn.Module):
    def __init__(self, config):
        super(RotatedYoloBase, self).__init__()
        self.config = config
        ##variables which does not need gradient
        S, B, A = self.config['S'], self.config['B'], self.config['A']
        angle_mult = self.config['angle_mult']

        anchors = get_anchors(self.config)
        self.anchors_scale = Variable(torch.from_numpy(np.reshape(anchors, [1, 1, B, 1, 2])).float(),
                                      requires_grad=False)
        angle_anchor = np.arange(0, A * angle_mult, angle_mult)
        self.angle_anchors = Variable(torch.from_numpy(np.reshape(angle_anchor, [1, 1, 1, A])).float(),
                                      requires_grad=False)
        self.S_wh_scale = Variable(torch.from_numpy(np.reshape([S, S], [1, 1, 1, 1, 2])).float(), requires_grad=False)

        if self.config['use_cuda']:
            self.anchors_scale = self.anchors_scale.cuda()
            self.S_wh_scale = self.S_wh_scale.cuda()
            self.angle_anchors = self.angle_anchors.cuda()

        self.features_all = self.get_features()
        self.regions = self.get_regions()
        self.classifier = self.get_classifier()

    def get_features(self):
        pass
    def get_regions(self):
        pass
    def get_classifier(self):
        pass

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.features_all(x)
        x = self.regions(x)
        x = self.classifier(x)

        B, A = self.config['B'], self.config['A']

        x = x.view((batch_size, -1, B, A, 6))

        confidence = x[:, :, :, :, 0]
        centerxy = x[:, :, :, :, 1:3]
        wh = x[:, :, :, :, 3:5]
        angle = x[:, :, :, :, 5]

        #####post processing
        confidence = F.sigmoid(confidence)
        # offset with respect to the grid cell
        centerxy = F.sigmoid(centerxy)
        # ratio w and h
        wh = torch.exp(wh) * self.anchors_scale
        # angle for each bbx
        angle = F.tanh(angle)
        ######

        return confidence, centerxy, wh, angle

    def get_angle_degree(self, angle_pred):
        angle_pred = torch.asin(angle_pred)
        angle_pred = angle_pred * 180 / 3.141593 + self.angle_anchors
        return angle_pred

    def calculate_loss(self, predictions, batch):
        lambda_noobj = self.config['lambda_noobj']
        lambda_obj = self.config['lambda_obj']
        lambda_coor = self.config['lambda_coor']
        loss_type = self.config['loss_type']
        S, B, A = self.config['S'], self.config['B'], self.config['A']

        conf_pred, centerxy_pred, wh_pred, angle_pred = predictions

        confs_mask = self.get_best_boxes(batch, centerxy_pred, wh_pred, angle_pred)

        # take care of the weight terms
        center_wh_angle_pred = torch.cat((centerxy_pred, torch.sqrt(wh_pred), angle_pred.unsqueeze(-1)), -1)

        center_wh_sz = center_wh_angle_pred.size()[-1]
        center_wh_wt = torch.cat(center_wh_sz * [confs_mask.unsqueeze(-1)], -1)
        center_wh_wt = lambda_coor * center_wh_wt

        batch_center_wh_angle = torch.cat((batch.center_wh, batch.angle.unsqueeze(-1)), -1)
        if loss_type == 'l2':
            loss_centerxy = torch.nn.MSELoss(center_wh_angle_pred, batch_center_wh_angle)
        else:
            loss_centerxy = F.smooth_l1_loss(center_wh_angle_pred, batch_center_wh_angle) 
            
        loss_centerxy = loss_centerxy * center_wh_wt
        loss_centerxy = torch.reshape(loss_centerxy, [-1, S * S * B * A * center_wh_sz])
        loss_centerxy = torch.sum(loss_centerxy, 1) / torch.sum(confs_mask)

        # only calculate conf of the selected anchor bounding box
        conf_wt = lambda_noobj * (1. - confs_mask) + lambda_obj * confs_mask
        if loss_type == 'l2':
            loss_obj = torch.nn.MSELoss(conf_pred, confs_mask) 
        else:
            loss_obj = F.smooth_l1_loss(conf_pred, confs_mask) 
            
        loss_obj = loss_obj * conf_wt
        loss_obj = torch.reshape(loss_obj, [-1, S * S * B * A])
        loss_obj = torch.sum(loss_obj, 1) / torch.sum(confs_mask)

        loss = loss_centerxy + loss_obj
        loss = .5 * torch.mean(loss)
        # print loss_obj, loss_centerxy
        return loss

    def get_best_boxes(self, batch, centerxy_pred, wh_pred, angle_pred):
        S, B, A = self.config['S'], self.config['B'], self.config['A']
        ariou_threshold = self.config['ariou_threshold']
        ########################################
        # pick the anchor box predicted having highest iou
        # this should be max rotated iou
        ########################################
        # calculate best iou predicted
        # project the wh on S grid
        wh_pred_S = wh_pred * self.S_wh_scale
        area_pred_S = wh_pred_S[:, :, :, :, 0] * wh_pred_S[:, :, :, :, 1]
        upleft_pred_S = centerxy_pred - (wh_pred_S * .5)
        botright_pred_S = centerxy_pred + (wh_pred_S * .5)

        # calculate the intersection areas (remember the indices runs from top to bottom and left to right
        intersect_upleft_S = torch.max(upleft_pred_S, batch.upleft)
        intersect_botright_S = torch.min(botright_pred_S, batch.botright)
        intersect_wh_S = intersect_botright_S - intersect_upleft_S
        intersect_wh_S = torch.nn.Threshold(0.0, 0.0)(intersect_wh_S)
        intersect_area_S = intersect_wh_S[:, :, :, :, 0] * intersect_wh_S[:, :, :, :, 1]

        # calculate the best IOU, set 0.0 confidence for worse boxes
        union_area = batch.areas + area_pred_S - intersect_area_S
        iou = intersect_area_S / union_area

        angle_pred = self.get_angle_degree(angle_pred)

        agnle_iou = torch.cos(angle_pred - batch.angle).abs()
        ariou = iou * agnle_iou
        ariou = torch.nn.Threshold(ariou_threshold, 0.0)(ariou)
        ariou = torch.reshape(ariou, (-1, S * S, B * A))
        # pick max iou bounding box cells
        max_ariou, _ = torch.max(ariou, dim=2, keepdim=True)
        best_box = torch.eq(ariou, max_ariou).float()
        best_box = torch.reshape(best_box, (-1, S * S, B, A))

        # pick only false negative, ignore false positive bounding boxes (don't care bbx)
        # loss_obj will take care of false positive object dtection
        confs_mask = best_box * batch.confs

        return confs_mask

class RotatedYoloSmall(RotatedYoloBase):
    def __init__(self, config):
        super(RotatedYoloSmall, self).__init__(config)

    def get_features(self):
        features = vgg16_bn(True).features
        modules = list(features.children())
        return nn.Sequential(*modules)

    def get_regions(self):
        layers = []
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def get_classifier(self):
        B, A = self.config['B'], self.config['A']

        return nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, B * A * (1 + 2 + 2 + 1), kernel_size=1),
        )

class RotatedYoloLarge(RotatedYoloBase):
    def __init__(self, config):
        super(RotatedYoloLarge, self).__init__(config)

    def get_features(self):
        features = vgg16_bn(True).features
        modules = list(features.children())
        return nn.Sequential(*modules)

    def get_regions(self):
        layers = []

        conv2d = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(1024), nn.LeakyReLU(inplace=True)]

        conv2d = nn.Conv2d(1024, 512, kernel_size=1)
        layers += [conv2d, nn.BatchNorm2d(512), nn.LeakyReLU(inplace=True)]

        conv2d = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(1024), nn.LeakyReLU(inplace=True)]

        conv2d = nn.Conv2d(1024, 512, kernel_size=1)
        layers += [conv2d, nn.BatchNorm2d(512), nn.LeakyReLU(inplace=True)]

        conv2d = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(1024), nn.LeakyReLU(inplace=True)]

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def get_classifier(self):
        B, A = self.config['B'], self.config['A']

        return nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, B * A * (1 + 2 + 2 + 1), kernel_size=1),
        )
