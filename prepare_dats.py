import cv2
import numpy as np
import os
import random
import shutil
import resource
import glob
resource.setrlimit(resource.RLIMIT_NOFILE, (3100,3100)) # Allow for enough files to be open

SHUF_ONCE = True
SHUF_TWICE = True

SHUF_DIR = '../data/shuf/'
DAT_DIR = '../data/dats/'
NF = 1000

ISIZE = 64
LSIZE = [7,7]
RSIZE = ISIZE*ISIZE*3+LSIZE[0]+LSIZE[1]

SHOW_IMAGE = False

def get_files(folder):
  files = glob.glob(folder+'*')
  print("{0}: {1} files total".format(folder, len(files)))
  files = [f for f in files if (int(f.split('/')[-1][:3]) < 100)]
  print("{0}: {1} files kept".format(folder, len(files)))
  files.sort()
  return files

def get_label_from_name(name):
  label = np.zeros((1,LSIZE[0]+LSIZE[1]))[0]
  c = int(name.split('/')[-1][9]) - 1
  m = LSIZE[0] + int(name.split('/')[-1][13])
  label[c] = 1
  label[m] = 1
  return label

def show_image(d):
  img = d[0,0:ISIZE*ISIZE*3]
  print(img.min(), img.max())
  cv2.imshow('img', img.reshape([ISIZE,ISIZE,3]).astype(np.uint8))
  print(d[0,ISIZE*ISIZE*3:])
  cv2.waitKey(1000)

live_dir = get_files('../data/' + 'live/')
fake_dir = get_files('../data/' + 'fake/*/')
fake_dir = [_ for _ in fake_dir if '_3_' in _.split('/')[-2]]

print(len(live_dir))
print(len(fake_dir))

random.shuffle(live_dir)
random.shuffle(fake_dir)
live_dir = live_dir[:int(len(live_dir)/4)]
fake_dir = fake_dir[:int(4*len(fake_dir)/6)]

print(len(live_dir))
print(len(fake_dir))

files = live_dir + fake_dir
random.shuffle(files)

if SHUF_ONCE:
  if os.path.isdir(SHUF_DIR):
    shutil.rmtree(SHUF_DIR)
  os.makedirs(SHUF_DIR)
  shuf_files = [open(SHUF_DIR + str(i+1) + '.dat', 'wb') for i in list(range(0,NF))]
  # Start shuffling
  for i,f in enumerate(files):
    print("{0}: {1}/{2}".format(f, i+1, len(files)))
    vid = cv2.VideoCapture(f)
    label = get_label_from_name(f)
    print(label)
    skipfr_i = 0
    while skipfr_i <= 250:
      suc, img = vid.read()
      if not suc:
        break
      if (skipfr_i % 10) != 0:
        skipfr_i+=1
        continue
      else:
        skipfr_i+=1

      img_dim = np.shape(img)

      # Extract a random region from the image
      for i in range(20):
        idx = int(random.randint(0,img_dim[1]-ISIZE))
        idy = int(random.randint(0,img_dim[0]-ISIZE))
        im = img[idy:idy+ISIZE,idx:idx+ISIZE,:]
        im = im.astype(np.uint8)
        #im = np.flip(im, 2)
        #im = np.transpose(im, (2,0,1))
        fi = random.randint(0,NF-1)
        shuf_files[fi].write(im.tobytes())
        shuf_files[fi].write(label.astype(np.uint8).tobytes())

      #cv2.imshow('', im.astype(np.uint8))
      #cv2.waitKey(1)

  for f in shuf_files:
    f.close()

if SHUF_TWICE:
  if os.path.isdir(DAT_DIR):
    shutil.rmtree(DAT_DIR)
  os.makedirs(DAT_DIR)
  for _bfi in range(1,NF+1):
    print('\r{0:3.1f}%'.format(100 * _bfi / NF), end='')
    bf = open('{0}{1}.dat'.format(SHUF_DIR, _bfi), 'rb')
    d = np.fromfile(bf, np.uint8)
    bf.close()

    d = d.reshape((-1, RSIZE))
    np.random.shuffle(d)
    d = np.ascontiguousarray(d)
    
    if SHOW_IMAGE:
      show_image(d)
    
    bf = open('{0}{1}.dat'.format(DAT_DIR, _bfi), 'wb')
    bf.write(d.astype(np.uint8))
    bf.close()
  print()

print('Done')
