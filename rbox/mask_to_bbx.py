import numpy as np
import pandas as pd
import cv2
from data_utils import rle_decode
def split_train_val():
    train_masks = pd.read_csv('input/train_ship_segmentations_bbox.csv')
    empty_img = masks[masks['EncodedPixels'].isnull()].ImageId.unique()
    non_empty_img = masks[~masks['EncodedPixels'].isnull()].ImageId.unique()
    
    val_non_empty_img_id = np.random.choice(non_empty_img, len(non_empty_img)/10, replace=False)
    val_empty_img_id = np.random.choice(empty_img, len(empty_img)/10, replace=False)

    train_non_empty_img_id = set(non_empty_img) - set(val_non_empty_img_id)
    train_empty_img_id = set(empty_img) - set(val_empty_img_id)

    np.save('input/val_non_empty_img_id.npy', np.array(list(val_non_empty_img_id)))
    np.save('input/val_empty_img_id.npy', np.array(list(val_empty_img_id)))
    np.save('input/train_non_empty_img_id.npy', np.array(list(train_non_empty_img_id)))
    np.save('input/train_empty_img_id.npy', np.array(list(train_empty_img_id)))

    print len(val_non_empty_img_id), len(train_non_empty_img_id), len(val_non_empty_img_id)+ len(train_non_empty_img_id), \
    len(val_empty_img_id) , len(train_empty_img_id), len(val_empty_img_id) + len(train_empty_img_id)

def create_rbox(in_filename, out_filename):
    masks = pd.read_csv('/home/atul/rotated_yolo/input/' + in_filename)
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

    masks.to_csv('/home/atul/rotated_yolo/input/' + out_filename, index=False)

if __name__ == '__main__':
    create_rbox('test_ship_segmentations.csv', 'test_ship_segmentations_bbox.csv')
    create_rbox('train_ship_segmentations.csv', 'train_ship_segmentations_bbox.csv')