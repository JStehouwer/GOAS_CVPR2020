import cv2
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from networks import GOGen
from database import Database

CONFIG = {
  'TRAIN_DIR':  './model_gogen/',
  'LIB_DIR':    './lib_gogen/',
  'ISIZES':     np.copy([64,64,3]).astype(np.int32),
  'LSIZES':     np.copy([7,7]).astype(np.int32),
  'MSIZES':     np.copy([32,32]).astype(np.int32),
  'GPU_TO_USE': '0',
  }

CONFIG['GPU_OPTIONS'] = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1, visible_device_list =CONFIG['GPU_TO_USE'], allow_growth = True)
CONFIG['GPU_CONFIG'] = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options = CONFIG['GPU_OPTIONS'])

if os.path.isdir(CONFIG['LIB_DIR']):
  shutil.rmtree(CONFIG['LIB_DIR'])
os.makedirs(CONFIG['LIB_DIR'])

G = tf.Graph()
S = tf.compat.v1.Session(graph=G, config=CONFIG['GPU_CONFIG'])

with G.as_default():
  with S.as_default():
    MODEL = GOGen(G, S, CONFIG['TRAIN_DIR'], CONFIG['ISIZES'], CONFIG['LSIZES'], CONFIG['MSIZES'], False)
    
    images = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 3), name='input')
    images = tf.image.resize(images, [64,64]) / 256
    images = tf.stack([images], axis=0)
    labels_c = tf.compat.v1.placeholder(tf.float32, shape=(CONFIG['LSIZES'][0]), name='label_c')
    labels_c = tf.reshape(labels_c, [1,CONFIG['LSIZES'][0]])
    labels_m = tf.compat.v1.placeholder(tf.float32, shape=(CONFIG['LSIZES'][1]), name='label_m')
    labels_m = tf.reshape(labels_m, [1,CONFIG['LSIZES'][1]])
    MODEL.run(images, labels_c, labels_m)

    img = tf.identity(MODEL.i_out, name='img')
    labc = tf.nn.softmax(MODEL.c_out, name='labc')
    labm = tf.nn.softmax(MODEL.m_out, name='labm')
    labcg = tf.nn.softmax(MODEL.cg_out, name='labcg')
    labmg = tf.nn.softmax(MODEL.mg_out, name='labmg')

    last_epoch = max(0, MODEL.load_if_exists())
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(CONFIG['LIB_DIR'])
    builder.add_meta_graph_and_variables(S, [tf.saved_model.SERVING])
    builder.save()
print('Model frozen and written to {0}'.format(CONFIG['LIB_DIR']))

