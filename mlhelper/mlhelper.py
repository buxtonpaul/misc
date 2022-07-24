from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from PIL import Image
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


def do_multiclass_prediction(model, filename, classnames, resizing=None, rescale=1.0):
    """
    Import an image from filename,
    makes a prediction and plots the image with the predicted class as title
    """
    # todo, rework this to use
    #     img = mpimg.imread(target_sample)
    # plt.imshow(img)
    # plt.title(f"Original Image from classs {target_class}")
    # plt.axis("Off")

    # augmented_image = data_augmentation(tf.expand_dims(img,axis=0),training=True)
    # plt.figure()
    # plt.imshow(tf.squeeze(augmented_image)/255.0)
    # resizing can be done with a tensorflow layer prior to the inference as well!

    img = Image.open(filename)

    if resizing is not None:
        img = img.resize((224, 224))
    plt.imshow(img)

    if resizing is not None:
        img = np.asarray(img).reshape((1,) + resizing + (3,))
    res = model.predict(img)
    idx = tf.argmax(res, axis=1)[0]
    classname = classnames[idx]

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
            plt.title(dat_in.class_names[tf.argmax(labels[i])])
            plt.axis("off")


def fetch_url(url):
    """
    Fetches a file using wget unless it already exists
    """
    pass


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
      history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Function to evaluate: accuracy, precision, recall, f1-score


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def wget(url):
    '''
    Downloads the file specified in the url in the current folder and returns the filename
    '''
    import requests
    # url = 'https://pypi.python.org/packages/source/g/guppy/' + fname
    elements = url.split('/')
    fname = elements[-1]
    r = requests.get(url)
    open(fname , 'wb').write(r.content)
    return fname
  