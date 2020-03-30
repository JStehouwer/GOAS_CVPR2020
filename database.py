import cv2
import glob
import numpy as np
import random
import tensorflow as tf

class Database():
  def __init__(self, path, isizes, lsizes, bs):
    self.path = path
    self.isizes = isizes
    self.lsizes = lsizes
    self.bs = bs
    self.filenames = glob.glob(self.path+'*.dat')
    self.ibytes = self.isizes[0] * self.isizes[1] * self.isizes[2]
    self.rbytes = self.ibytes + self.lsizes[0] + self.lsizes[1]
    self.dataset = self.get_dataset()
    self.images, self.c_labels, self.m_labels = self.dataset.get_next()

  def parse(self, data):
    data = tf.io.decode_raw(data, tf.uint8)
    
    img = tf.reshape(tf.strided_slice(data, [0], [self.ibytes]), self.isizes)
    img = tf.cast(img, tf.float32) / 255
    img.set_shape(self.isizes)
    
    clab = tf.reshape(tf.strided_slice(data, [self.ibytes], [self.ibytes+self.lsizes[0]]), [self.lsizes[0]])
    clab = tf.cast(clab, tf.float32)
    clab.set_shape([self.lsizes[0]])
    
    mlab = tf.reshape(tf.strided_slice(data, [self.ibytes+self.lsizes[0]], [self.rbytes]), [self.lsizes[1]])
    mlab = tf.cast(mlab, tf.float32)
    mlab.set_shape([self.lsizes[1]])
    
    return img, clab, mlab

  def get_dataset(self):
    dataset = tf.data.FixedLengthRecordDataset(self.filenames, self.rbytes)
    dataset = dataset.repeat(-1) # Repeat indefinitely
    dataset = dataset.map(self.parse) # Parse the elements
    dataset = dataset.batch(batch_size=self.bs) # Batchify the data
    dataset = tf.compat.v1.data.make_one_shot_iterator(dataset) # Get an iterator for the dataset
    return dataset


