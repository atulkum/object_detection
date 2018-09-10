import cv2
import numpy as np

def box_iou_rotated(centerxy1, wh1, angle1, centerxy2, wh2, angle2):
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


def box_iou_axis_aligned(centerxy1, wh1, centerxy2, wh2):
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

    iou = intersection_area / (w1 * h1 + w2 * h2 - intersection_area)
    return iou


def nms(config, conf, centerxy, wh, angle):
    threshold = config['conf_threshold']
    nms_threshold = config['nms_threshold']

    sorted_conf = np.flip(np.argsort(conf), axis=0)
    discarded_indices = set()

    pred_length = len(sorted_conf)
    for i1 in range(pred_length):
        idx1 = sorted_conf[i1]
        # only consider boundin boxes having conf > threshold
        if conf[idx1] <= threshold: break
        # is idx1 is already discarded continue
        if idx1 in discarded_indices: continue

        for i2 in range(i1 + 1, pred_length):
            idx2 = sorted_conf[i2]
            if conf[idx2] <= threshold: break
            if idx2 in discarded_indices: continue

            if box_iou_rotated(centerxy[idx1], wh[idx1], angle[idx1],
                                    centerxy[idx2], wh[idx2], angle[idx2]) > nms_threshold:
                discarded_indices.add(idx2)

    boxes = list()
    for idx in sorted_conf:
        if idx not in discarded_indices and conf[idx] > threshold:
            x, y = centerxy[idx]
            w, h = wh[idx]

            boxes.append((x, y, w, h, angle[idx], conf[idx]))

    return boxes

if __name__ == '__main__':
    #<type 'tuple'>: ((59.553284, 407.77975), (63.709957, 59.200657), 27.584908)
    #<type 'tuple'>: ((94.235535, 381.20895), (12.19889, 16.471848), -71.986496)

    print box_iou_rotated((59.553284, 407.77975), (63.709957, 59.200657),27.584908, (94.235535, 381.20895), (12.19889, 16.471848),  -71.986496)