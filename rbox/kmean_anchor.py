import numpy as np
import pandas as pd
import os
import sys

def IOU(x,centroids):
    similarities = []
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities)

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(n):
        sum+= max(IOU(X[i],centroids))
    return sum/n

def kmeans(X, centroids, eps, anchor_file):
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)
    ''' 
    orig_img_sz = 768.
    S = 7.
    for i in range(anchors.shape[0]):
        anchors[i][0] *= orig_img_sz / S
        anchors[i][1] *= orig_img_sz / S
    '''

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    f.write('"anchors" : [')
    for i in sorted_indices:
        f.write('{"w": %0.2f, "h":%0.2f}, ' % (anchors[i, 0], anchors[i, 1]))

    f.write('],\n')
    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    masks = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_bbox.csv'))
    img_ids = np.load(os.path.join(data_dir, 'val_non_empty_img_id.npy'))
    img_ids = np.concatenate((img_ids, np.load(os.path.join(data_dir, 'train_non_empty_img_id.npy'))))
    objs = []
    for img_id in img_ids:
        w = masks.loc[masks['ImageId'] == img_id, 'w'].tolist()
        h = masks.loc[masks['ImageId'] == img_id, 'h'].tolist()

        for obj in zip(*[w, h]):
            objs.append(obj)

    annotation_dims = np.array(objs)

    eps = 0.005
    for num_clusters in range(1, 11):
        anchor_file = os.path.join(output_dir, 'anchors%d.txt' % (num_clusters))

        indices = np.random.choice(np.arange(annotation_dims.shape[0]), num_clusters, replace=False)
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file)
        print('centroids.shape', centroids.shape)
