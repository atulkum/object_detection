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

    if ret == 2:
        return 1.
    elif ret == 0:
        return 0.
    else:
        intersection_area = cv2.contourArea(poly)

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

    indices = []
    for i in range(len(sorted_conf)):
        idx = sorted_conf[i]
        if conf[idx] <= threshold: break
        
        keep = True;
        for k in range(len(indices)): 
            if not keep:
                break
            kept_idx = indices[k];
            overlap = box_iou_rotated(centerxy[idx], wh[idx], angle[idx],
                                    centerxy[kept_idx], wh[kept_idx], angle[kept_idx]) 
            keep = (overlap <= threshold)
        
        if keep:
            indices.append(idx);
            
    boxes = list()
    for idx in indices:
        x, y = centerxy[idx]
        w, h = wh[idx]

        boxes.append((x, y, w, h, angle[idx], conf[idx]))
    
    return boxes
