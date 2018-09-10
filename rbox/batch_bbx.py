import numpy as np
import traceback
import torch
from torch.autograd import Variable
from im_transform import imcv2_affine_trans, imcv2_recolor
from torchvision import transforms
import cv2
from collections import defaultdict

class Batch(object):
    def __init__(self, config, all_img_objs, img_ids):
        self.img_ids = img_ids
        self.config = config
        self.to_tensor = transforms.ToTensor()
        self.whiten_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        A = self.config['A']
        angle_mult = self.config['angle_mult']
        self.angle_anchor = np.arange(0, A * angle_mult, angle_mult)

        all_vars = defaultdict(list)

        for img, all_obj in all_img_objs:
            im, trans_param = self.img_preprocess(img)     
            all_vars['img'].append(im)

            if all_obj is not None:
                confs, center_wh, angle, upleft, botright, areas = self.get_bbx_regressor(all_obj, trans_param)
                all_vars['confs'].append(torch.from_numpy(confs))
                all_vars['center_wh'].append(torch.from_numpy(center_wh))
                all_vars['angle'].append(torch.from_numpy(angle))
                all_vars['upleft'].append(torch.from_numpy(upleft))
                all_vars['botright'].append(torch.from_numpy(botright))
                all_vars['areas'].append(torch.from_numpy(areas))

        self.to_vars(all_vars)

    def __len__(self):
        return len(self.img_ids)

    def to_vars(self, all_vars):
        for k in all_vars:
            v = all_vars[k]
            v = Variable(torch.stack(v).float())
            if self.config['use_cuda']:
                v = v.cuda()
            setattr(self, k, v)

    def resize_input(self, im):
        in_size = self.config['in_size']
        imsz = cv2.resize(im, (in_size, in_size))
        return imsz

    def img_preprocess(self, im):
        trans_param = None
        if self.config["is_img_aug"]:
            im, _, trans_param = imcv2_affine_trans(im)
            im = imcv2_recolor(im)

        im = self.resize_input(im)
        im = self.to_tensor(im)
        im = self.whiten_img(im)

        return im, trans_param

    def bbx_preprocess(self, obj, trans_param):
        orig_l = self.config["orig_img_size"]
        x, y, wb, hb, a = obj

        # angle between the longer side and vertical
        offset = 180 if wb < hb else 90
        a += offset
        if trans_param is not None:
            scale, offs, flip = trans_param
            xoff, yoff = offs
            x = x * scale - xoff
            x = max(min(x, orig_l), 0)
            y = y * scale - yoff
            y = max(min(y, orig_l), 0)
            wb = min(wb * scale, orig_l)
            hb = min(hb * scale, orig_l)
            if flip:
                # only consider horizontal flip
                x = orig_l - x
                a = 180 - a

        return x, y, wb, hb, a

    def get_bbx_regressor(self, allobj, trans_param):
        S, B, A = self.config['S'], self.config['B'], self.config['A']
        orig_l = self.config["orig_img_size"]
        angle_mult = self.config["angle_mult"]

        angle_anchor = np.arange(0, A * angle_mult, angle_mult)

        confs = np.zeros([S * S, B, A])
        center_wh = np.zeros([S * S, B, A, 4])
        angle = np.zeros([S * S, B, A])
        prear = np.zeros([S * S, 4])

        for j, obj in enumerate(allobj):
            x, y, wb, hb, a = self.bbx_preprocess(obj, trans_param)

            cx = (x / orig_l) * S
            cy = (y / orig_l) * S

            #offset inside the grid
            x = cx - np.floor(cx)
            y = cy - np.floor(cy)

            #proportion of size with respect to whole image
            wb = wb / orig_l
            hb = hb / orig_l

            #caching for iou calculation
            # which grid cell the center fall into
            center_offset = int(np.floor(cy) * S + np.floor(cx))

            prear[center_offset, 0] = x - .5 * wb * S # xleft
            prear[center_offset, 1] = y - .5 * hb * S # yup
            prear[center_offset, 2] = x + .5 * wb * S # xright
            prear[center_offset, 3] = y + .5 * hb * S # ybot

            wb = np.sqrt(wb)
            hb = np.sqrt(hb)

            center_wh[center_offset, :, :, :] = [[[x, y, wb, hb]] * A] * B
            confs[center_offset, :, :] = [[1.] * A] * B
            #angle offset
            angle_offset = a - angle_anchor
            angle_offset = (angle_offset / 180) * 3.141593
            angle_offset = np.sin(angle_offset)

            angle[center_offset, :, :] = [angle_offset] * B

        upleft = np.expand_dims(prear[:, 0:2], 1)
        botright = np.expand_dims(prear[:, 2:4], 1)
        wh = botright - upleft
        areas = wh[:, :, 0] * wh[:, :, 1]

        upleft = np.concatenate([upleft] * B * A, 1).reshape((S*S, B, A, 2))
        botright = np.concatenate([botright] * B * A, 1).reshape((S*S, B, A, 2))
        areas = np.concatenate([areas] * B * A, 1).reshape((S*S, B, A))

        return confs, center_wh, angle, upleft, botright, areas

