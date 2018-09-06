import Queue
from threading import Thread
import time
import numpy as np
import traceback
from data_utils import get_img, get_gen_img_mask, get_bbx
from batch_bbx import Batch

class AbstractGenerator(object):
    def __init__(self, config, masks):
        self.config = config
        self.masks = masks

    def get_img_bbx(self, img_id):
        img = get_img(self.config, img_id)
        all_obj = get_bbx(img_id, self.masks)
        return img, all_obj

    def get_gen_img_bbox(self, empty_img_id, non_empty_img_id):
        img, _ = get_gen_img_mask(self.config, empty_img_id, non_empty_img_id, self.masks)
        all_obj = get_bbx(non_empty_img_id, self.masks)
        return img, all_obj

    def get_one_batch(self):
        pass

class RandomGenerator(AbstractGenerator):
    def __init__(self, empty_ids, nonempty_ids, config, masks):
        super(RandomGenerator, self).__init__(config, masks)
        self.empty_ids = empty_ids
        self.nonempty_ids = nonempty_ids

    def create_batch(self, augmented = False):
        all_img_objs = []
        all_img_ids = []
        batch_size = self.config['batch_size']

        non_empty_n = batch_size/2 if augmented else batch_size
        non_empty_id = np.random.choice(self.nonempty_ids, non_empty_n, replace=False)
        for idx in non_empty_id:
            all_img_ids.append(idx)
            img, all_obj = self.get_img_bbx(idx)
            all_img_objs.append((img, all_obj))

        if augmented:
            aug_id_empty = np.random.choice(self.empty_ids, batch_size / 2, replace=False)
            aug_id_nonempty = np.random.choice(self.nonempty_ids, batch_size / 2, replace=False)

            for i in range(len(aug_id_empty)):
                all_img_ids.append('%s-%s'%(aug_id_empty[i], aug_id_nonempty[i]))
                img, all_obj = self.get_gen_img_bbox(aug_id_empty[i], aug_id_nonempty[i])
                all_img_objs.append((img, all_obj))

        return Batch(self.config, all_img_objs, all_img_ids)

    def get_one_batch(self):
        while True:
            yield self.create_batch()

class SequenceGenerator(AbstractGenerator):
    def __init__(self, empty_ids, nonempty_ids, config, masks):
        super(SequenceGenerator, self).__init__(config, masks)
        self.img_ids = np.concatenate((empty_ids, nonempty_ids))
        self.masked_set = nonempty_ids

    def create_batch(self, idx):
        batch_size = self.config['batch_size']
        all_img_objs = []
        all_img_ids = []
        for img_id in self.img_ids[idx: idx + batch_size]:
            if img_id in self.masked_set:
                img, all_obj = self.get_img_bbx(img_id)
            else:
                img = get_img(self.config, img_id)
                all_obj = []
            all_img_ids.append(img_id)
            all_img_objs.append((img, all_obj))

        return Batch(self.config, all_img_objs, all_img_ids)

    def get_one_batch(self):
        batch_size = self.config['batch_size']
        for i in range(0, len(self.img_ids), batch_size):
            yield self.create_batch(i)

class TestSequenceGenerator(AbstractGenerator):
    def __init__(self, img_ids, config):
        super(TestSequenceGenerator, self).__init__(config, None)
        self.img_ids = img_ids

    def create_batch(self, idx):
        batch_size = self.config['batch_size']
        all_img_objs = []
        all_img_ids = []
        for img_id in self.img_ids[idx: idx + batch_size]:
            img = get_img(self.config, img_id)
            all_obj = None
            all_img_ids.append(img_id)
            all_img_objs.append((img, all_obj))

        return Batch(self.config, all_img_objs, all_img_ids)

    def get_one_batch(self):
        batch_size = self.config['batch_size']
        for i in range(0, len(self.img_ids), batch_size):
            yield self.create_batch(i)

class Batcher(object):
    BATCH_QUEUE_MAX = 10
    def __init__(self, generator):
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self.generator = generator
        single_pass = not isinstance(generator, RandomGenerator)
        if single_pass:
            self._num_batch_q_threads = 1
        else:
            self._num_batch_q_threads = 4

        self._batch_q_threads = []
        for _ in xrange(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        batch = self._batch_queue.get()
        return batch

    def fill_batch_queue(self):
        for batch in self.generator.get_one_batch():
            self._batch_queue.put(batch)
        self.shutdown()

    def shutdown(self):
        self._batch_queue.put(None)

    def watch_threads(self):
        while True:
            time.sleep(60)
            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

def main():
    import pandas as pd
    import json
    nonemp = ['002fdcf51.jpg',	'6d948c270.jpg',	'6d97350bf.jpg',	'6d9833913.jpg',	'6d98c508a.jpg',	'6d9b9be19.jpg',	'6d9d3ed34.jpg',	'6d9e5af16.jpg']
    emp = []
    masks = pd.read_csv('../input/train_ship_segmentations_bbox.csv')
    config = json.load(open('/Users/atulkumar/Downloads/work/kggpy/config1.json'))
    gen = SequenceGenerator(emp, nonemp, config, masks)
    batcher = Batcher(gen)
    i = 0

    while not gen.is_finished():
        a = batcher.next_batch()
        if a is None:
            break
        print i, '==' , a.coord
        i += 1

if __name__ == '__main__':
    ''' 
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print "Shutdown requested...exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)
    '''

    import cv2

    r1 = ((122.5, 239.5), (5, 5), -30)
    r2 = ((122, 239), (4, 4), 0)

    ret, poly = cv2.rotatedRectangleIntersection(r1, r2)

    #ret = 0/1/2, none/partial/full
    print ret, cv2.contourArea(poly)

    #############
    #draw_rectangle(None, im, allobj, '/Users/atulkumar/Downloads/work/kggpy/log', True)
    ############