from __future__ import division
from math import sqrt as sqrt
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from data_utils import get_anchors, get_angle_anchors
from iou_utils import get_ariou_torch
import torch.nn.init as init
from itertools import product as product

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
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
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
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
        angle_anchor = get_angle_anchors(self.config)
        
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
        #angle = F.tanh(angle)
        angle = F.sigmoid(angle)
        ######

        return confidence, centerxy, wh, angle

    def calculate_loss(self, predictions, batch):
        lambda_noobj = self.config['lambda_noobj']
        lambda_obj = self.config['lambda_obj']
        lambda_coor = self.config['lambda_coor']
        loss_type = self.config['loss_type']
        S, B, A = self.config['S'], self.config['B'], self.config['A']

        conf_pred, centerxy_pred, wh_pred_S, angle_pred = predictions

        confs_mask = self.get_best_boxes(batch, centerxy_pred, wh_pred_S, angle_pred)

        # take care of the weight terms
        center_wh_angle_pred = torch.cat((centerxy_pred, torch.sqrt(wh_pred_S), angle_pred.unsqueeze(-1)), -1)

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

    def get_best_boxes(self, batch, centerxy_pred, wh_pred_S, angle_pred):
        S, B, A = self.config['S'], self.config['B'], self.config['A']
        ariou_threshold = self.config['ariou_threshold']
        ########################################
        # pick the anchor box predicted having highest iou
        # this should be max rotated iou
        ########################################
        # calculate best iou predicted
        ariou = get_ariou_torch(wh_pred_S, centerxy_pred, angle_pred, batch, self.angle_anchors)
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

        #layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def get_classifier(self):
        B, A = self.config['B'], self.config['A']

        return nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, B * A * (1 + 2 + 2 + 1), kernel_size=1),
        )


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class RotatedSSD():
    def __init__(self, config):
        vgg_layers = []
        in_channels = 3
        for v in cfg['D_300']:
            if v == 'M':
                vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                vgg_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        features = nn.Sequential(*vgg_layers)
        features.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        vgg_layers = list(features.children())

        vgg_layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        vgg_layers += [conv6, nn.ReLU(inplace=True)]
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        vgg_layers += [conv7, nn.ReLU(inplace=True)]

        extra_layers = []
        in_channels = 1024
        kernel_size = [1,3]
        kernel_size_idx = 0
        extra_300 = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        for k, v in enumerate(extra_300):
            if in_channels != 'S':
                if v == 'S':
                    extra_layers += [nn.Conv2d(in_channels, extra_300[k + 1],
                                         kernel_size=kernel_size[kernel_size_idx], stride=2, padding=1)]
                else:
                    extra_layers += [nn.Conv2d(in_channels, v, kernel_size=kernel_size[kernel_size_idx])]
                kernel_size_idx = (kernel_size_idx + 1) % 2
            in_channels = v

        self.num_regressor = 6
        output_layers = []
        vgg_source = [21, -2]
        multi_boxes = [4, 6, 6, 6, 4, 4]
        for k, v in enumerate(vgg_source):
            output_layers += [nn.Conv2d(vgg_layers[v].out_channels,
                                                multi_boxes[k] * self.num_regressor, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            output_layers += [nn.Conv2d(v.out_channels,
                                                multi_boxes[k] * self.num_regressor, kernel_size=3, padding=1)]

        self.vgg = nn.ModuleList(vgg_layers)
        self.extras = nn.ModuleList(extra_layers)
        self.output = nn.ModuleList(output_layers)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

    def get_prior(self):
        mean = []
        feature_maps = [38, 19, 10, 5, 3, 1]
        image_size = 300
        steps = [8, 16, 32, 64, 100, 300]
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        

        for k, f in enumerate(feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = image_size / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = min_sizes[k]/image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (max_sizes[k]/image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.output):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)


        regressor = loc.view(loc.size(0), -1, self.num_regressor)

        return regressor


