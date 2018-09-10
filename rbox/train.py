import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from train_utils import create_dirs, get_model, write_summary
from batcher_bbx import Batcher, RandomGenerator, SequenceGenerator

logging.basicConfig(level=logging.INFO)

class Train(object):
    def __init__(self, config_file):
        self.config = json.load(open(config_file))

        self.read_dataset(self.config['root_dir'])
        self.model_dir, self.bestmodel_path, self.summary_writer = create_dirs(self.config['root_dir'], config_file)
    
    def read_dataset(self, root_dir):
        data_dir = os.path.join(root_dir, 'input')
        self.val_non_empty_img_id = np.load(os.path.join(data_dir, 'val_non_empty_img_id.npy'))
        self.val_empty_img_id = np.load(os.path.join(data_dir,'val_empty_img_id.npy'))
        self.train_non_empty_img_id = np.load(os.path.join(data_dir,'train_non_empty_img_id.npy'))
        self.train_empty_img_id = np.load(os.path.join(data_dir, 'train_empty_img_id.npy'))
        #####
        '''
        nonemp = ['002fdcf51.jpg', '6d948c270.jpg', '6d97350bf.jpg', '6d9833913.jpg', '6d98c508a.jpg', '6d9b9be19.jpg',
                  '6d9d3ed34.jpg', '6d9e5af16.jpg']

        self.train_non_empty_img_id = nonemp
        self.train_empty_img_id = []

        self.val_non_empty_img_id = nonemp
        self.val_empty_img_id = []
        '''
        ######
        self.masks = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_bbox.csv'))
        
    def save_model(self, exp_loss, iter):
        logging.info("Saving to %s..." % self.bestmodel_path)
        state = {
            'iter': iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_exp_loss': exp_loss
        }
        torch.save(state, self.bestmodel_path)

    def eval_one_batch(self, batch):
        predictions = self.model(batch.img)
        loss = self.model.calculate_loss(predictions, batch)
        return loss.item()

    def eval_all(self):
        val_gen = SequenceGenerator(self.train_empty_img_id ,
                                    self.train_non_empty_img_id,
                                    self.config,
                                    self.masks)
        val_batcher = Batcher(val_gen)
        test_loss = 0.0
        test_num = 0.0
        self.model.eval()
        batch = val_batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)
            test_num  += len(batch)
            test_loss += loss * len(batch)

            batch = val_batcher.next_batch()

        if test_num > 0:
            return test_loss/test_num
        else:
            return -1

    def setup_train(self, model_name, model_file_path=None):
        #batcher
        train_gen = RandomGenerator(self.train_empty_img_id ,
                                    self.train_non_empty_img_id,
                                    self.config,
                                    self.masks)
        self.train_batcher = Batcher(train_gen)

        #model
        self.model = get_model(self.config)
        params = self.model.parameters()
        req_params = filter(lambda p: p.requires_grad, self.model.parameters())
        logging.info("Number of params: %d Number of params required grad: %d" % (sum(p.numel() for p in params),
                                                                                  sum(p.numel() for p in req_params)))
        #optimizer
        initial_lr = self.config['lr']
        self.optimizer = optim.Adam(req_params, lr=initial_lr)

        start_iter, start_loss = 0, 0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.model.load_state_dict(state['state_dict'])

            start_iter = state['iter']
            start_loss = state['current_exp_loss']
           
            self.optimizer.load_state_dict(state['optimizer'])
            if self.config['use_cuda']:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self):
        batch = self.train_batcher.next_batch()
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(batch.img)
        loss = self.model.calculate_loss(predictions, batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def trainIters(self, model_file_path):
        iter, exp_loss = self.setup_train(model_file_path)
        start = time.time()
        best_dev_loss = None

        while iter < self.config['n_iters']:
            iter += 1
            loss = self.train_one_batch()
                
            exp_loss = 0.99 * exp_loss + 0.01 * loss if not exp_loss else loss
            write_summary(loss, "train/loss", self.summary_writer, iter)
            
            if iter % 100 == 0:
                self.summary_writer.flush()
                
            if iter % self.config['print_every'] == 0:
                logging.info('Iter %d, seconds for %d batch: %.2f , loss: %f' % (iter, self.config['print_every'],
                                                                           time.time() - start, exp_loss))
                start = time.time()

            if iter % self.config['eval_every'] == 0:
                dev_loss = self.eval_all()
                logging.info("Iter %d, Dev loss: %f" % (iter, dev_loss))

                write_summary(dev_loss, "dev/loss",  self.summary_writer, iter)
                
                if best_dev_loss is None or dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    self.save_model(exp_loss, iter)

if __name__ == '__main__':
    config_file = sys.argv[1]
    model_file_path = None
    if len(sys.argv) > 2:
        train_dir = sys.argv[2]
        config_file = os.path.join(train_dir, 'config.json')
        model_dir = os.path.join(train_dir, 'model')
        model_file_path = os.path.join(model_dir, 'bestmodel')
        
    train_processor = Train(config_file)
    train_processor.trainIters(model_file_path)
