import numpy as np
from iou_utils import box_iou_rotated

def nms(config, conf, centerxy, wh, angle):
    threshold = config['conf_threshold']
    nms_threshold = config['nms_threshold']
    
    sorted_conf = np.flip(np.argsort(conf), axis=0)

    indices = []
    for i in range(len(sorted_conf)):
        idx = sorted_conf[i]
        if conf[idx] <= threshold: break
        
        keep = True
        for k in range(len(indices)): 
            if not keep:
                break
            kept_idx = indices[k]
            overlap = box_iou_rotated(centerxy[idx], wh[idx], angle[idx],
                                    centerxy[kept_idx], wh[kept_idx], angle[kept_idx]) 
            keep = (overlap < nms_threshold)
        
        if keep:
            indices.append(idx)
            
    boxes = list()
    for idx in indices:
        x, y = centerxy[idx]
        w, h = wh[idx]

        boxes.append((x, y, w, h, angle[idx], conf[idx]))
    
    return boxes
