import numpy as np
import pandas as pd
import cv2
from data_utils import rle_decode

def create_rbox():
    masks = pd.read_csv('../input/train_ship_segmentations_bbox.csv')
    shape = (768, 768)
    all_masks = masks['EncodedPixels'].tolist()
    xa = []
    ya = []
    wa = []
    ha = []
    aa = []

    x_bbx = []
    y_bbx = []
    w_bbx = []
    h_bbx = []

    for maski in all_masks:
        if pd.isnull(maski):
            xa.append(-1)
            ya.append(-1)
            wa.append(-1)
            ha.append(-1)
            aa.append(-1)

            x_bbx.append(-1)
            y_bbx.append(-1)
            w_bbx.append(-1)
            h_bbx.append(-1)

            continue

        maski_img = np.zeros(shape)
        maski_img += rle_decode(maski, shape)
        maski_img = maski_img.astype(np.uint8)
        _, contours, _ = cv2.findContours(maski_img, 1, 1)
        (x, y), (w, h), a = cv2.minAreaRect(contours[0])

        xa.append(x)
        ya.append(y)
        wa.append(w)
        ha.append(h)
        aa.append(a)

        x, y, w, h = cv2.boundingRect(maski_img)
        x_bbx.append(x)
        y_bbx.append(y)
        w_bbx.append(w)
        h_bbx.append(h)


    masks['x'] = xa
    masks['y'] = ya
    masks['w'] = wa
    masks['h'] = ha
    masks['a'] = aa

    masks['x_bbx'] = x_bbx
    masks['y_bbx'] = y_bbx
    masks['w_bbx'] = w_bbx
    masks['h_bbx'] = h_bbx

    masks.to_csv('../input/train_ship_segmentations_bbox.csv', index=False)

if __name__ == '__main__':
    create_rbox()