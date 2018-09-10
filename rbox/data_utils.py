import numpy as np 
import pandas as pd
from skimage.data import imread
import traceback
import os
import cv2
import copy

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def full_rle_decode(img_mask, shape):
    all_masks = np.zeros(shape)
    if pd.isnull(img_mask[0]):
        return all_masks
    for mask in img_mask:
        all_masks += rle_decode(mask, shape)
    return all_masks

def get_img(config, img_id):
    try:
        if isinstance(config, str):
            image_data_dir = config
        else:
            image_data_dir = os.path.join(config['root_dir'], 'input/train')
        return imread(os.path.join(image_data_dir, img_id))
    except:
        print traceback.format_exc()
        return None

def get_mask_img(config, img_id, masks):
    img = get_img(config, img_id)
    if img is None:
        return None, None
    shape = img.shape
    img_masks = masks.loc[masks['ImageId'] == img_id, 'EncodedPixels'].tolist()

    all_masks = full_rle_decode(img_masks, shape)
    return img, all_masks

def get_gen_img_mask(dir_path, empty_img_id, non_empty_img_id, masks):
    img = get_img(dir_path, empty_img_id)
    if img is None:
        return None
    img_ne, all_masks = get_mask_img(dir_path, non_empty_img_id, masks)
    if img_ne is None:
        return None
    shape = img.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if all_masks[i,j] == 1:
                img[i, j] = img_ne[i, j]
    return img

def get_anchors(config):
    anchors = []
    orig_img_size = config['orig_img_size']
    for anchor in config['anchors']:
        wb = anchor['w'] / float(orig_img_size)
        hb = anchor['h'] / float(orig_img_size)
        anchors.append((wb, hb))

    return np.array(anchors)

def get_angle_anchors(config):
    angle_mult = config["angle_mult"]
    A = config["A"]
        
    if angle_mult == 0:
        angle_anchor = np.array([0.0])
    else:
        angle_anchor = np.arange(0, A * angle_mult, angle_mult)
    return angle_anchor

def get_bbx(img_id, masks):
    if masks is None:
        return []
    x = masks.loc[masks['ImageId'] == img_id, 'x'].tolist()
    y = masks.loc[masks['ImageId'] == img_id, 'y'].tolist()
    w = masks.loc[masks['ImageId'] == img_id, 'w'].tolist()
    h = masks.loc[masks['ImageId'] == img_id, 'h'].tolist()
    a = masks.loc[masks['ImageId'] == img_id, 'a'].tolist()

    return zip(*[x, y, w, h, a])

def get_bbx_no_angle(img_id, masks):
    if masks is None:
        return []
    x = masks.loc[masks['ImageId'] == img_id, 'x_bbx'].tolist()
    y = masks.loc[masks['ImageId'] == img_id, 'y_bbx'].tolist()
    w = masks.loc[masks['ImageId'] == img_id, 'w_bbx'].tolist()
    h = masks.loc[masks['ImageId'] == img_id, 'h_bbx'].tolist()
    a = [0.0]*len(h)

    return zip(*[x, y, w, h, a])


def draw_rectangle(image_path, img, bboxes, out_dir, angle_offset=False):
    if not isinstance(img, str):
        imgcv = copy.deepcopy(img)
        img_id = "out.jpg"
    else:
        img_id = img
        imgcv = get_img(image_path, img_id)
    if img is None:
        return
    
    w_orig, h_orig, _ = imgcv.shape

    for x, y, w, h, angle in bboxes:
        if angle_offset:
            offset = 180 if w < h else 90
            angle += offset

        rrect = ((x, y), (w, h), angle)
        box = cv2.boxPoints(rrect)
        box = np.int0(box)
        cv2.drawContours(imgcv, [box], 0, (0, 255, 0), 2)
        # fillConvexPoly

    img_file = os.path.join(out_dir, img_id)
    cv2.imwrite(img_file, imgcv)