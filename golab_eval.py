import cv2
import math
import numpy as np
import os
import random
import scipy
import shutil
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed(123)

skipfr = 10

read_dir = '../data/'
write_dir = './results/'

model_dir = './lib/'

ISIZE = 64
PERFRAME = 10

if os.path.isdir(write_dir):
  shutil.rmtree(write_dir)
os.makedirs(write_dir)

# Load the FACEPAD model
sess = tf.compat.v1.Session()
tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], model_dir)
image = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
labelc = tf.compat.v1.get_default_graph().get_tensor_by_name("labc:0")
labelm = tf.compat.v1.get_default_graph().get_tensor_by_name("labm:0")

def get_files(result, path, directory):
  filetypes = ['.mp4', '.mov', '.MOV']
  files = os.listdir(path + directory)
  for f in files: # For all files
    if int(f[0:3]) > 100:
        for t in filetypes: # For all supported filetypes
          if t in f: # If supported filetype
            result.append(directory + f)
  return result

def get_filelist(path):
  result = []
  result = get_files(result, path, 'live/')
  result = get_files(result, path, 'fake/')
  result.sort()
  return result

def process(img):
  labc, labm = sess.run([labelc, labelm],feed_dict={image: im})
  return labc[0], labm[0]

videos = get_filelist(read_dir)
for vi, video in enumerate(videos):
  print('Video: `' + video + '` (' + str(vi) + '/' + str(len(videos)) + ')')
  t_filename = write_dir + '.'.join(video.replace('/', '_').split('.')[0:-1]) + '.txt'
  if os.path.isfile(t_filename):
    continue
  vid = cv2.VideoCapture(read_dir + video)
  skipfr_i = 0
  f = open(t_filename, 'w')
  while True:
    suc, img = vid.read()
    if not suc:
      break
    if (skipfr_i % skipfr) != 0:
      skipfr_i+=1
      continue
    else:
      skipfr_i+=1

    img_dim = np.shape(img)

    # Extract a random region from the image
    for i in range(PERFRAME):
      idx = int(math.floor(random.random() * (img_dim[1] - ISIZE)))
      idy = int(math.floor(random.random() * (img_dim[0] - ISIZE)))
      im = img[idy:idy+ISIZE,idx:idx+ISIZE,:]

      # Process the region
      labc_, labm_ = process(im)
      
      st = ' '.join(map(str, np.round(labc_, 5))) + ' ' + ' '.join(map(str, np.round(labm_, 5))) + '\n'
      f.write(st)
    #cv2.imshow('', im.astype(np.uint8))
    #cv2.waitKey(1)
