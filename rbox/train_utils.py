import importlib
import tensorflow as tf
import os
import time
import shutil

def create_dirs(root_dir, config_file):
    log_dir = os.path.join(root_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train_dir = os.path.join(log_dir, 'train_%d' % (int(time.time())))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    model_dir = os.path.join(train_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    shutil.copy(config_file, train_dir)

    bestmodel_path = os.path.join(model_dir, 'bestmodel')
    summary_writer = tf.summary.FileWriter(train_dir)

    return model_dir, bestmodel_path, summary_writer

def get_model(model_name, config):
    module = importlib.import_module('rolo')
    class_ = getattr(module, model_name)
    model = class_(config)
    if config['use_cuda']:
        model = model.cuda()
    return model

def write_summary(value, tag, summary_writer, global_step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


