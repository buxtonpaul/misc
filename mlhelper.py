import tensorflow as tf
import datetime
import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import zipfile
import pathlib


# unzip
def unzip_data(filename):
    """
    Unzips filename into current dir
    """
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall()
    zip_ref.close()


# visualise images


def view_random_image(target_dir, target_class):
    """
    Displays a random image for the folder 'target_dir' subfolder 'target_class'
    Returns the image
    """
    target_folder = target_dir + target_class
    # random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # read in the image and plot using matplotlib
    img = mpimg.imread(target_folder + '/' + random_image[0])

    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    return img


def create_tensorboard_callback(experiment_name, dir_name="logs/fit"):
    """
    Create a tensorboard callback function to store logs for
    'experiment_name' under the 'dir_name' folder
    Returns the callback to be passed into model.fit
    """
    log_dir = dir_name + "/" + experiment_name + "/" + \
        datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    print(f"saving Tensorboard logfiles to {log_dir}")
    return tensorboard_callback


def do_multiclass_prediction(model, filename, classnames, resizing=None):
    """
    Import an image from filename, 
    makes a prediction and plots the image with the predicted class as title
    """
    img = mpimg.imread(filename)
    if resizing is not None:
        img = img.resize(resizing)

    res = model.predict(tf.expand_dims(img, 0))
    idx = tf.argmax(res, axis=1)
    classname = classnames[tuple(idx)]

    plt.imshow(img)
    plt.title(classname)
    plt.axis("off")


def view_dataset(dat_in):
    """
    Plots a set of inputs from a tensorflow dataset
    """
    plt.figure(figsize=(10, 10))
    for images, labels in dat_in.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dat_in.class_names[labels[i]])
            plt.axis("off")
