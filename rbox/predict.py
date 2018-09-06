import json
import logging
import os
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from batcher_bbx import Batcher, TestSequenceGenerator
from data_utils import draw_rectangle
from rbox.rolo import RotatedYolo

logging.basicConfig(level=logging.INFO)

class Predict(object):
    def __init__(self, config_file, model_file_path):
        self.config = json.load(open(config_file))

        self.model = RotatedYolo(self.config)
        if self.config['use_cuda']:
            self.model = self.model.cuda()

        #state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        #self.model.load_state_dict(state['state_dict'])
        self.model.eval()

        orig_img_size = self.config['orig_img_size']
        self.orig_wh_scale = Variable(torch.from_numpy(np.reshape([orig_img_size, orig_img_size],
                                                                  [1, 1, 1, 1, 2])).float(), requires_grad=False)

    def box_iou_rotated(self, centerxy1, wh1, angle1, centerxy2, wh2, angle2):
        x1, y1 = centerxy1
        w1, h1 = wh1
        x2, y2 = centerxy2
        w2, h2 = wh2

        rrect1 = ((x1, y1), (w1, h1), angle1)
        rrect2 = ((x2, y2), (w2, h2), angle2)
        ret, poly = cv2.rotatedRectangleIntersection(rrect1, rrect2)

        if ret > 0:
            intersection_area = cv2.contourArea(poly)
        else:
            intersection_area = 0

        iou = intersection_area / (w1 * h1 + w2 * h2 - intersection_area)
        return iou

    def box_iou_axis_aligned(self, centerxy1, wh1, angle1, centerxy2, wh2, angle2):
        x1, y1 = centerxy1
        w1, h1 = wh1
        x2, y2 = centerxy2
        w2, h2 = wh2

        def overlap(x1, w1, x2, w2):
            l1 = x1 - w1 / 2.
            l2 = x2 - w2 / 2.
            left = max(l1, l2)
            r1 = x1 + w1 / 2.
            r2 = x2 + w2 / 2.
            right = min(r1, r2)
            return right - left

        w = overlap(x1, w1, x2, w2)
        h = overlap(y1, h1, y2, h2)

        if w < 0 or h < 0:
            intersection_area = 0
        else:
            intersection_area = w * h

        iou = intersection_area/(w1 * h1 + w2 * h2 - intersection_area)
        return iou

    def nms(self, conf, centerxy, wh, angle):
        threshold = self.config['conf_threshold']
        nms_threshold = self.config['nms_threshold']

        sorted_conf = np.flip(np.argsort(conf), axis=0)
        discarded_indices = set()

        pred_length = len(sorted_conf)
        for i1 in range(pred_length):
            idx1 = sorted_conf[i1]
            #only consider boundin boxes having conf > threshold
            if conf[idx1] <= threshold: break
            #is idx1 is already discarded continue
            if idx1 in discarded_indices: continue

            for i2 in range(i1 + 1, pred_length):
                idx2 = sorted_conf[i2]
                if conf[idx2] <= threshold: break
                if idx2 in discarded_indices: continue

                if self.box_iou_rotated(centerxy[idx1], wh[idx1], angle[idx1],
                                        centerxy[idx2], wh[idx2], angle[idx2]) > nms_threshold:
                    discarded_indices.add(idx2)

        boxes = list()
        for idx in sorted_conf:
            if idx not in discarded_indices and conf[idx] > threshold:
                x, y = centerxy[idx]
                w, h = wh[idx]

                boxes.append((x, y, w, h, angle[idx], conf[idx]))

        return boxes

    def find_boxes(self, img, img_ids):
        S = self.config['S']
        B = self.config['B']
        A = self.config['A']
        orig_img_size = self.config['orig_img_size']

        conf_pred, centerxy_pred, wh_pred, angle_pred = self.model(img)
        wh_pred_orig = wh_pred * self.orig_wh_scale

        angle_pred = self.model.get_angle_degree(angle_pred)

        conf_pred = conf_pred.data.numpy()
        centerxy_pred = centerxy_pred.data.numpy()
        wh_pred_orig = wh_pred_orig.data.numpy()
        angle_pred = angle_pred.data.numpy()

        cellxy = 1. * orig_img_size / S
        angle_pred = np.clip(angle_pred, 0.0, 180.0)

        boxes = {}
        for i, img_id in enumerate(img_ids):
            conf_pred_ = conf_pred[i]
            centerxy_pred_ = centerxy_pred[i].reshape((S, S, B, A, 2))
            wh_pred_orig_ = wh_pred_orig[i].reshape((S, S, B, A, 2))
            angle_pred_ = angle_pred[i].reshape((S, S, B, A))

            for row in range(S):
                for col in range(S):
                    for box_loop in range(B):
                        for angle_loop in range(A):
                            centerxy_pred_[row, col, box_loop, angle_loop, 0] = (col + centerxy_pred_[
                                row, col, box_loop, angle_loop, 0]) * cellxy
                            centerxy_pred_[row, col, box_loop, angle_loop, 1] = (row + centerxy_pred_[
                                row, col, box_loop, angle_loop, 1]) * cellxy

                            offset = 180 if wh_pred_orig_[row, col, box_loop, angle_loop, 0] < wh_pred_orig_[
                                row, col, box_loop, angle_loop, 1] else 90
                            angle_pred_[row, col, box_loop, angle_loop] -= offset

            all_boxes_len = S*S*B*A
            conf_pred_ = conf_pred_.reshape((all_boxes_len))
            centerxy_pred_ = centerxy_pred_.reshape((all_boxes_len, 2))
            wh_pred_orig_ = wh_pred_orig_.reshape((all_boxes_len, 2))
            angle_pred_ = angle_pred_.reshape((all_boxes_len))
            boxes[img_id] = self.nms(conf_pred_, centerxy_pred_, wh_pred_orig_, angle_pred_)

        return boxes

    def dump_result(self, img_ids, image_path):
        out_dir = os.path.join(image_path, 'out_%d' % (int(time.time())))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        test_gen = TestSequenceGenerator(img_ids, self.config)
        test_batcher = Batcher(test_gen)

        batch = test_batcher.next_batch()
        while batch is not None:
            boxes = self.find_boxes(batch.img, batch.img_ids)
            for img_id in boxes:
                self.dump_result_single_batch(img_id, boxes[img_id], image_path, out_dir)

            batch = test_batcher.next_batch()

    def dump_result_single_batch(self, img_id, single_img_boxes, image_path, out_dir):
        resultsForJSON = []
        bboxes = []
        for b in single_img_boxes:
            x, y, w, h, angle, conf = b
            bboxes.append((x, y, w, h, angle))
            resultsForJSON.append({"confidence": float(conf), "angle": float(angle),
                                   "center": {"x": float(x), "y": float(y)}, "size": {"w": float(w), "h": float(h)}})

        draw_rectangle(image_path, img_id, bboxes, out_dir)

        textJSON = json.dumps(resultsForJSON)
        text_file_name = os.path.splitext(img_id)[0] + ".json"
        text_file = os.path.join(out_dir, text_file_name)
        with open(text_file, 'w') as f:
            f.write(textJSON)

if __name__ == '__main__':
    predictor = Predict('/Users/atulkumar/Downloads/work/kggpy/config2.json',
                        '/Users/atulkumar/Downloads/work/kggpy/log/train_1536138488/model/bestmodel')
    img_ids = ['002fdcf51.jpg', '6d948c270.jpg', '6d97350bf.jpg', '6d9833913.jpg', '6d98c508a.jpg', '6d9b9be19.jpg',
     '6d9d3ed34.jpg', '6d9e5af16.jpg']


    predictor.dump_result(img_ids, "/Users/atulkumar/Downloads/work/kggpy/input/train")
