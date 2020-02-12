#!/usr/bin/env python
# coding: utf-8

# ## Set up the input pipeline

# In[1]:


import tensorflow as tf


# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import IPython.display as display
import pathlib
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[3]:


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


# In[4]:


data_dir_testA = '/Users/taylor/Google Drive/Developer/computer_vision/datasets/VizslaDalmatian/testA/'
data_dir_trainA = '/Users/taylor/Google Drive/Developer/computer_vision/datasets/VizslaDalmatian/trainA/'
data_dir_testB = '/Users/taylor/Google Drive/Developer/computer_vision/datasets/VizslaDalmatian/testB/'
data_dir_trainB = '/Users/taylor/Google Drive/Developer/computer_vision/datasets/VizslaDalmatian/trainB/'


# In[5]:


CLASS_NAMES_testA = np.array(['testA'])
CLASS_NAMES_trainA = np.array(['trainA'])
CLASS_NAMES_testB = np.array(['testB'])
CLASS_NAMES_trainB = np.array(['trainB'])


# In[6]:


list_ds_testA = tf.data.Dataset.list_files(str(data_dir_testA+'*/*'))
list_ds_trainA = tf.data.Dataset.list_files(str(data_dir_trainA+'*/*'))
list_ds_testB = tf.data.Dataset.list_files(str(data_dir_testB+'*/*'))
list_ds_trainB = tf.data.Dataset.list_files(str(data_dir_trainB+'*/*'))


# In[7]:


for f in list_ds_trainB.take(5):
  print(f.numpy())


# In[8]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path_testA(file_path):
  label = 'testA'
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
def process_path_trainA(file_path):
  label = 'trainA'
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
def process_path_testB(file_path):
  label = 'testB'
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
def process_path_trainB(file_path):
  label = 'trainB'
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


# In[9]:


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds_testA = list_ds_testA.map(process_path_testA, num_parallel_calls=AUTOTUNE)
labeled_ds_trainA = list_ds_trainA.map(process_path_trainA, num_parallel_calls=AUTOTUNE)
labeled_ds_testB = list_ds_testB.map(process_path_testB, num_parallel_calls=AUTOTUNE)
labeled_ds_trainB = list_ds_trainB.map(process_path_trainB, num_parallel_calls=AUTOTUNE)


# In[10]:


train_vizslas = labeled_ds_trainA
test_vizslas = labeled_ds_testA
train_dalmatians = labeled_ds_trainB
test_dalmatians = labeled_ds_testB


# In[11]:


shuffle_buffer_size=1000
train_vizslas = train_vizslas.shuffle(buffer_size=shuffle_buffer_size)
train_dalmatians = train_dalmatians.shuffle(buffer_size=shuffle_buffer_size)


# In[12]:


def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image


# In[13]:


# normalizing the images to [-1, 1]
def normalize(image):
  #image = tf.cast(image, tf.float32)
  image = image - 1.0
  return image


# In[14]:


def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image


# In[15]:


def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


# In[16]:


def preprocess_image_test(image, label):
  image = normalize(image)
  return image


# In[17]:


train_vizslas = train_vizslas.map(
    preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_dalmatians = train_dalmatians.map(
    preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_vizslas = test_vizslas.map(
    preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_dalmatians = test_dalmatians.map(
    preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)


# In[18]:


sample_vizslas = next(iter(train_vizslas))
sample_dalmatians = next(iter(train_dalmatians))


# In[19]:


plt.subplot(121)
plt.title('Visla')
plt.imshow(sample_vizslas[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Vizsla with random perturbation')
plt.imshow(random_jitter(sample_vizslas[0]) * 0.5 + 0.5)


# In[20]:


plt.subplot(121)
plt.title('Dalmatian')
plt.imshow(sample_dalmatians[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Dalmatian with random jitter')
plt.imshow(random_jitter(sample_dalmatians[0]) * 0.5 + 0.5)


# In[21]:


OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# In[1]:


to_dalmatians = generator_g(sample_vizslas)
to_vizslas = generator_f(sample_dalmatians)
plt.figure(figsize=(8, 8))
contrast = 2

imgs = [sample_vizslas, to_dalmatians, sample_dalmatians, to_vizslas]
title = ['Vizsla', 'To Dalmatian', 'Dalmatian', 'To Vizsla']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()


# In[23]:


plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real Dalmatian?')
plt.imshow(discriminator_y(sample_dalmatians)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real Vizsla?')
plt.imshow(discriminator_x(sample_vizslas)[0, ..., -1], cmap='RdBu_r')

plt.show()


# ## Loss functions

# In[24]:


LAMBDA = 10


# In[25]:


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[26]:


def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5


# In[27]:


def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


# In[28]:


def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1


# In[29]:


def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


# Initialize the optimizers for all the generators and the discriminators.

# In[30]:


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# ## Checkpoints

# In[31]:


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  #ckpt.restore('C:/Users/Cole Geisler/Documents/Taylor/tensorFlow/VizslaDalmatian2/checkpoints/train/ckpt-4')
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


# ## Training
# 
# Note: This example model is trained for fewer epochs (40) than the paper (200) to keep training time reasonable for this tutorial. Predictions may be less accurate. 

# In[32]:


EPOCHS = 0


# In[33]:


def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


# * Get the predictions.
# * Calculate the loss.
# * Calculate the gradients using backpropagation.
# * Apply the gradients to the optimizer.

# In[34]:


@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


# In[35]:


for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_vizslas, train_dalmatians)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n+=1

  clear_output(wait=True)

  generate_images(generator_g, sample_vizslas)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))


# ## Generate using test dataset

# In[36]:


# Run the trained model on the test dataset
for inp in test_vizslas:
  generate_images(generator_g, inp)


# In[ ]:




