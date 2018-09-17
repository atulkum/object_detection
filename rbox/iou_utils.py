import cv2
import torch

def get_angle_degree_torch(angle_pred, angle_anchors):
    angle_pred = torch.asin(angle_pred)
    angle_pred = angle_pred * 180 / 3.141593 + angle_anchors
    return angle_pred

def box_iou_rotated(centerxy1, wh1, angle1, centerxy2, wh2, angle2):
    x1, y1 = centerxy1
    w1, h1 = wh1
    x2, y2 = centerxy2
    w2, h2 = wh2

    if int(y2) == 387 or int(y1) == 387:
        pass

    rrect1 = ((x1, y1), (w1, h1), angle1)
    rrect2 = ((x2, y2), (w2, h2), angle2)
    ret, poly = cv2.rotatedRectangleIntersection(rrect1, rrect2)

    if ret == 2:
        return 1.
    elif ret == 0:
        return 0.
    else:
        intersection_area = cv2.contourArea(poly)

    #iou = intersection_area / (w1 * h1 + w2 * h2 - intersection_area)
    iou = intersection_area / min(w1 * h1, w2 * h2)
    return iou

def get_ariou_torch(wh_pred_S, centerxy_pred, angle_pred, batch, angle_anchors):
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

    angle_pred = get_angle_degree_torch(angle_pred, angle_anchors)

    agnle_iou = torch.cos(angle_pred - batch.angle).abs()
    ariou = iou * agnle_iou
    ariou = torch.clamp(ariou, min=0)
    return ariou

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

    #iou = intersection_area / (w1 * h1 + w2 * h2 - intersection_area)
    iou = intersection_area / min(w1 * h1, w2 * h2)
    return iou