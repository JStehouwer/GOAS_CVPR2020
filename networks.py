import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import os

batch_norm_params = {
  'decay': 0.9,
  'epsilon': 1e-5,
  'scale': True,
  'updates_collections': None,
}

def Conv(inp, nk, ks, name, training):
  batch_norm_params['is_training'] = training
  with arg_scope( [layers.conv2d],
    kernel_size = ks,
    weights_initializer = tf.random_normal_initializer(stddev=0.02),
    biases_initializer  = tf.constant_initializer(0.0),
    activation_fn=tf.nn.elu,
    normalizer_fn=layers.batch_norm,
    normalizer_params=batch_norm_params,
    trainable = training,
    padding='SAME',
    stride=1,
    reuse=tf.compat.v1.AUTO_REUSE):
    c = layers.conv2d(inp, num_outputs=nk, kernel_size=[ks,ks], scope=name)
  return c

def Pool(inp, name):
  p = layers.max_pool2d(inp, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope=name)
  return p

def Block(inp, nks, name, training, inp_sc=None, inp_sc_s=0):
  p0 = Pool(inp, name+'_pool0')
  if inp_sc is not None:
    p0 = tf.concat([tf.image.resize(inp_sc,inp_sc_s),p0], 3)
  
  c1 = Conv(p0, nks[0], 3, name+'_conv1', training)
  c2 = Conv(c1, nks[1], 3, name+'_conv2', training)
  c3 = Conv(c2, nks[2], 3, name+'_conv3', training)
  return c3

def SoftmaxBlock(inp, lsize, name, training):
  noise = Conv(inp, 3, 3, name+'_dnoise', training)
  rs0 = layers.flatten(noise)
  fc0 = layers.fully_connected(rs0, 512, activation_fn=None, reuse=tf.compat.v1.AUTO_REUSE, scope=name+'_dfc0')
  fc1 = layers.fully_connected(fc0, int(lsize), activation_fn=None, reuse=tf.compat.v1.AUTO_REUSE, scope=name+'_dfc1')
  return fc1

def Inference(images, lsizes, msizes, training=True):
  conv0 = Conv(images, 64, 5, 'd_conv0', training)
  
  blck1 = Block(conv0, [96,128,96], 'd_blck1', training)
  blck2 = Block(blck1, [128,156,128], 'd_blck2', training, inp_sc=images, inp_sc_s=msizes/2)
  blck3 = Block(blck2, [96,128,96], 'd_blck3', training, inp_sc=images, inp_sc_s=msizes/4)
  
  map1 = tf.image.resize(blck1,msizes)
  map2 = tf.image.resize(blck2,msizes)
  map3 = tf.image.resize(blck3,msizes)
  maps = tf.concat([map1,map2,map3], 3)
  
  conv1 = Conv(maps, 96, 3, 'd_conv1', training)
  conv2 = Conv(conv1, 64, 3, 'd_conv2', training)
  dp0 = layers.dropout(conv2, 0.5, scope='d_dp0')
  
  c_soft = SoftmaxBlock(dp0, lsizes[0], 'd_cam', training)
  m_soft = SoftmaxBlock(dp0, lsizes[1], 'd_med', training)
  
  return c_soft, m_soft

def Discriminator(images, msizes, training=True):
  conv0 = Conv(images, 32, 5, 'r_conv0', training)
  blck1 = Block(conv0, [32,64,64], 'r_blck1', training)
  blck2 = Block(conv0, [64,96,96], 'r_blck2', training)
  
  map1 = tf.image.resize(blck1,msizes)
  map2 = tf.image.resize(blck2,msizes)
  maps = tf.concat([map1, map2], 3)
  
  conv1 = Conv(maps, 64, 3, 'r_conv1', training)
  conv2 = Conv(conv1, 32, 3, 'r_conv2', training)
  dp0 = layers.dropout(conv2, 0.5, scope='r_dp0')
  
  rs0 = layers.flatten(dp0)
  fc1 = layers.fully_connected(rs0, 256, activation_fn=None, reuse=tf.compat.v1.AUTO_REUSE, scope='r_fc1')
  fc2 = layers.fully_connected(fc1, 2, activation_fn=None,reuse=tf.compat.v1.AUTO_REUSE, scope='r_fc2')
  
  return fc2

def choose_noise(labels, lablen, ims, name):
  labels = tf.stack([tf.stack([labels], axis=1)], axis=1)
  with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
    label_values = tf.compat.v1.get_variable(name, [1, ims, ims, lablen], initializer=tf.random_normal_initializer())
  labels = labels * label_values
  labels = tf.reduce_sum(labels, axis=3, keepdims=True)
  return labels

def Generator(images, clabels, clen, mlabels, mlen, ims, training):
  clabels = choose_noise(clabels, clen, ims, 'g_noise_c')
  mlabels = choose_noise(mlabels, mlen, ims, 'g_noise_m')
  images_noise = tf.concat([images, clabels, mlabels], axis=3)
  
  conv0 = Conv(images_noise, 64, 3, 'g_conv0', training)
  conv1 = Conv(conv0, 96, 3, 'g_conv1', training)
  conv2 = Conv(conv1, 96, 3, 'g_conv2', training)
  conv3 = Conv(conv2, 96, 3, 'g_conv3', training)
  conv4 = Conv(conv3, 96, 3, 'g_conv4', training)
  
  conc0 = tf.concat([clabels, mlabels, conv0, conv1, conv2, conv3, conv4], 3)
  conv5 = Conv(conc0, 64, 1, 'g_conv5', training)
  conv6 = Conv(conv5, 96, 1, 'g_conv6', training)
  conv7 = Conv(conv6, 64, 3, 'g_conv7', training)
  
  conc1 = tf.concat([clabels, mlabels, conv5, conv6, conv7], 3)
  conv8 = Conv(conc1, 3, 1, 'g_conv8', training)
  
  return images + conv8

def loss_softmax(gtl, ll):
  gtl = tf.stop_gradient(gtl)
  loss_per = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gtl, logits=ll)
  loss = tf.reduce_mean(loss_per)
  return loss * 10, loss_per

def loss_images(gtl, out):
  gtl = tf.stop_gradient(gtl)
  loss = tf.reduce_mean(tf.abs(gtl - out))
  return loss

class Network():
  def __init__(self, graph, sess, path, isizes, lsizes, msizes, training):
    self.graph = graph
    self.sess = sess
    self.path = path
    self.isizes = isizes
    self.lsizes = lsizes
    self.msizes = msizes
    self.lsizes = lsizes
    self.training = training
    self.saver = None

  def restore(self):
    last_epoch = -1
    ckpt = tf.train.get_checkpoint_state(self.path)
    if ckpt and ckpt.model_checkpoint_path:
      if self.saver is None:
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100,allow_empty=False)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      last_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('-')[-1])
    return last_epoch

  def load_if_exists(self):
    last_epoch = self.restore()
    if last_epoch > -1:
      print('Restore from Epoch {0}'.format(last_epoch))
    else:
      tf.compat.v1.global_variables_initializer().run()
      print('Train from scratch.')
    return last_epoch

  def save(self, _ei):
    if self.saver is None:
      self.saver = tf.compat.v1.train.Saver(max_to_keep=30,allow_empty=False)
    self.saver.save(self.sess, self.path + str(_ei))

class GOLab(Network):
  def __init__(self, graph, sess, path, isizes, lsizes, msizes, training):
    super().__init__(graph, sess, path, isizes, lsizes, msizes, training)

  def run(self, images, clabels, mlabels):
    self.c_out, self.m_out = Inference(images, self.lsizes, self.msizes, self.training)
    if self.training:
      l_c, _ = loss_softmax(clabels, self.c_out)
      l_m, _ = loss_softmax(mlabels, self.m_out)
      l = l_c + l_m
      self.loss = l
      self.losses = {'loss': l, 'l_c': l_c, 'l_m': l_m}

class GODisc(Network):
  def __init__(self, graph, sess, path, isizes, lsizes, msizes, training):
    super().__init__(graph, sess, path, isizes, lsizes, msizes, training)

class GOGen(Network):
  def __init__(self, graph, sess, path, isizes, lsizes, msizes, training):
    super().__init__(graph, sess, path, isizes, lsizes, msizes, training)

  def run(self, images, clabels, mlabels):
    if self.training:
      images_h = tf.split(images, 2, axis=0, num=2)
      clabels_h = tf.split(clabels, 2, axis=0, num=2)
      mlabels_h = tf.split(mlabels, 2, axis=0, num=2)
      
      # Perform Generator here
      images_c = Generator(images_h[0], clabels_h[1], self.lsizes[0], mlabels_h[1], self.lsizes[1], self.isizes[0], self.training)
      images_m = Generator(images_h[1], clabels_h[0], self.lsizes[0], mlabels_h[0], self.lsizes[1], self.isizes[0], self.training)
      
      # Perform Inference here
      ec, em = Inference(images, self.lsizes, self.msizes, self.training)
      ecc, ecm = Inference(images_c, self.lsizes, self.msizes, self.training)
      emc, emm = Inference(images_m, self.lsizes, self.msizes, self.training)
      
      # Perform Discriminator here
      er = Discriminator(images, self.msizes, self.training)
      erc = Discriminator(images_c, self.msizes, self.training)
      erm = Discriminator(images_m, self.msizes, self.training)
      
      # Compute loss for the GOLab
      dl_c, dl_c_p = loss_softmax(clabels, ec)
      dl_m, dl_m_p = loss_softmax(mlabels, em)
      self.dl = dl_c + dl_m
      
      # Compute loss for the GODisc
      rl_r, rl_r_p = loss_softmax(tf.concat([tf.ones([1,1]),tf.zeros([1,1])],-1), er)
      rl_s, rl_s_p = loss_softmax(tf.concat([tf.zeros([1,1]),tf.ones([1,1])],-1), tf.concat([erc, erm], 0))
      self.rl = rl_r + rl_s
      
      # Compute loss for the GOGen
      dl_c_p_h = tf.split(dl_c_p, 2, axis=0, num=2)
      dl_m_p_h = tf.split(dl_m_p, 2, axis=0, num=2)
      gl_cc, gl_cc_p = loss_softmax(clabels_h[0], ecc)
      gl_cc = tf.reduce_mean(gl_cc_p / tf.stop_gradient(dl_c_p_h[0] + 1))
      gl_cm, gl_cm_p = loss_softmax(mlabels_h[1], ecm)
      gl_cm = tf.reduce_mean(gl_cm_p / tf.stop_gradient(dl_m_p_h[1] + 1))
      gl_mc, gl_mc_p = loss_softmax(clabels_h[1], emc)
      gl_mc = tf.reduce_mean(gl_mc_p / tf.stop_gradient(dl_c_p_h[1] + 1))
      gl_mm, gl_mm_p = loss_softmax(mlabels_h[0], emm)
      gl_mm = tf.reduce_mean(gl_mm_p / tf.stop_gradient(dl_m_p_h[0] + 1))
      
      gl_r = loss_softmax(tf.concat([tf.ones([1,1]),tf.zeros([1,1])],-1), tf.concat([erc, erm], 0))
      gl_r = (1 / 8) * gl_r[0]
      gl_i = (255 / 50) * (loss_images(images_h[0], images_c) + loss_images(images_h[1], images_m))
      self.gl = gl_cc + gl_cm + gl_mc + gl_mm + gl_r + gl_i
      
      self.losses = {
        'dl': self.dl, 'dl_c': dl_c, 'dl_m': dl_m, # GOLab
        'rl': self.rl, # GODisc
        'gl': self.gl, 'gl_cc': gl_cc, 'gl_cm': gl_cm, 'gl_mc': gl_mc, 'gl_mm': gl_mm, # GOGen
        'gl_r': gl_r, 'gl_i': gl_i # GOGen GAN Loss
        }
    else:
       images_g = Generator(images, clabels, self.lsizes[0], mlabels, self.lsizes[1], self.isizes[0], self.training)
       ec, em = Inference(images, self.lsizes, self.msizes, self.training)
       ec_g, em_g = Inference(images_g, self.lsizes, self.msizes, self.training)
       self.i_out = images_g
       self.c_out = ec
       self.m_out = em
       self.cg_out = ec_g
       self.mg_out = em_g

