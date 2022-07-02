import zipfile
import pathlib


#unzip
def unzip_data(filename):
  """ 
  Unzips filename into current dir
  """
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()
  
  
# visualise images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

def view_random_image(target_dir, target_class):
  """
  Displays a random image for the folder 'target_dir' subfolder 'target_class'
  Returns the image
  """
  target_folder = target_dir  + target_class
  # random image path
  random_image = random.sample(os.listdir(target_folder),1)

  # read in the image and plot using matplotlib
  img = mpimg.imread(target_folder + '/' + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  return img

import tensorflow as tf
import datetime
def create_tensorboard_callback(experiment_name,dir_name="logs/fit"):
  """
  Create a tensorboard callback function to store logs for
  'experiment_name' under the 'dir_name' folder
  Returns the callback to be passed into model.fit
  """
  log_dir = dir_name + "/" + experiment_name +"/" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  print(f"saving Tensorboard logfiles to {log_dir}")
  return tensorboard_callback
