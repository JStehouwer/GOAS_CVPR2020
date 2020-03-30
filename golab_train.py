import cv2
from datetime import datetime
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from networks import GOLab
from database import Database

CONFIG = {
  'TRAIN_DIR':       './model/',
  'TRAIN_DATA':      '../data/dats/',
  'ISIZES':          np.copy([64,64,3]).astype(np.int32),
  'LSIZES':          np.copy([7,7]).astype(np.int32),
  'MSIZES':          np.copy([32,32]).astype(np.int32),
  'MAX_EPOCHS':      100,
  'STEPS_PER_EPOCH': 5000,
  'GPU_TO_USE':      '0',
  'BATCH_SIZE':      160,
  'LEARNING_RATE':   0.001
  }

CONFIG['GPU_OPTIONS'] = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1, visible_device_list =CONFIG['GPU_TO_USE'], allow_growth = True)
CONFIG['GPU_CONFIG'] = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options = CONFIG['GPU_OPTIONS'])

G = tf.Graph()
S = tf.compat.v1.Session(graph=G, config=CONFIG['GPU_CONFIG'])

def print_time():
  return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with G.as_default():
  with S.as_default():
    DATABASE = Database(CONFIG['TRAIN_DATA'], CONFIG['ISIZES'], CONFIG['LSIZES'], CONFIG['BATCH_SIZE'])
    
    MODEL = GOLab(G, S, CONFIG['TRAIN_DIR'], CONFIG['ISIZES'], CONFIG['LSIZES'], CONFIG['MSIZES'], True)
    
    MODEL.run(DATABASE.images, DATABASE.c_labels, DATABASE.m_labels)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    OPTIM = tf.compat.v1.train.AdamOptimizer(CONFIG['LEARNING_RATE']).minimize(MODEL.loss, global_step=global_step)

def run_step(_si):
  run_dict = MODEL.losses
  run_dict['optim'] = OPTIM
  res = S.run(run_dict)
  res.pop('optim', None)
  loss_str = ', '.join(['{0} {1:.3f}'.format(l, res[l]) for l in res])
  if _si % 10 == 0:
    print('\r{0} - Step {1} - {2}'.format(print_time(), _si, loss_str)+' '*10, end='')
  if _si % 500 == 0:
    print('')

def run_epoch(_ei):
  print('Epoch {0}'.format(_ei))
  for _si in range(0, CONFIG['STEPS_PER_EPOCH']):
    run_step(_si)
  MODEL.save(_ei+1)
  print()

with G.as_default():
  with S.as_default():
    last_epoch = max(0, MODEL.load_if_exists())
    for _ei in range(last_epoch, CONFIG['MAX_EPOCHS']):
      run_epoch(_ei)

