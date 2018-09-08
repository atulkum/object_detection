import json
import logging
import os
import time
import torch
import numpy as np

from batcher_bbx import Batcher, TestSequenceGenerator
from data_utils import draw_rectangle
from nms import nms
from train_utils import get_model

logging.basicConfig(level=logging.INFO)

class Predict(object):
    def __init__(self, config_file, model_file_path, model_name):
        self.config = json.load(open(config_file))
        self.model = get_model(model_name, self.config)
        state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['state_dict'])
        self.model.eval()

    def find_boxes(self, img, img_ids):
        S, B, A = self.config['S'], self.config['B'], self.config['A']
        orig_img_size = self.config['orig_img_size']

        conf_pred, centerxy_pred, wh_pred, angle_pred = self.model(img)

        angle_pred = self.model.get_angle_degree(angle_pred)

        conf_pred = conf_pred.data.numpy()
        centerxy_pred = centerxy_pred.data.numpy()
        wh_pred = wh_pred.data.numpy()
        angle_pred = angle_pred.data.numpy()

        wh_pred = wh_pred * orig_img_size
        cellxy = 1. * orig_img_size / S
        angle_pred = np.clip(angle_pred, 0.0, 180.0)

        boxes = {}
        for i, img_id in enumerate(img_ids):
            conf_pred_ = conf_pred[i]
            centerxy_pred_ = centerxy_pred[i].reshape((S, S, B, A, 2))
            wh_pred_ = wh_pred[i].reshape((S, S, B, A, 2))
            angle_pred_ = angle_pred[i].reshape((S, S, B, A))

            for row in range(S):
                for col in range(S):
                    for box_loop in range(B):
                        for angle_loop in range(A):
                            centerxy_pred_[row, col, box_loop, angle_loop, 0] = (col + centerxy_pred_[
                                row, col, box_loop, angle_loop, 0]) * cellxy
                            centerxy_pred_[row, col, box_loop, angle_loop, 1] = (row + centerxy_pred_[
                                row, col, box_loop, angle_loop, 1]) * cellxy

                            offset = 180 if wh_pred_[row, col, box_loop, angle_loop, 0] < wh_pred_[
                                row, col, box_loop, angle_loop, 1] else 90
                            angle_pred_[row, col, box_loop, angle_loop] -= offset

            all_boxes_len = S*S*B*A
            conf_pred_ = conf_pred_.reshape((all_boxes_len))
            centerxy_pred_ = centerxy_pred_.reshape((all_boxes_len, 2))
            wh_pred_ = wh_pred_.reshape((all_boxes_len, 2))
            angle_pred_ = angle_pred_.reshape((all_boxes_len))
            boxes[img_id] = nms(self.config, conf_pred_, centerxy_pred_, wh_pred_, angle_pred_)

        return boxes

    def dump_result(self, img_ids, image_path, out_path):
        out_dir = os.path.join(out_path, 'out_%d' % (int(time.time())))
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
                        '/Users/atulkumar/Downloads/work/kggpy/log/train_1536360690/model/bestmodel',
                        'RotatedYoloSmall')
    img_ids = ['6d9b9be19.jpg']
    #['002fdcf51.jpg', '6d948c270.jpg', '6d97350bf.jpg', '6d9833913.jpg', '6d98c508a.jpg', '6d9b9be19.jpg', '6d9d3ed34.jpg', '6d9e5af16.jpg']


    predictor.dump_result(img_ids, "/Users/atulkumar/Downloads/work/kggpy/input/train", "/Users/atulkumar/Downloads/work/kggpy/log")
