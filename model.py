import csv
import math
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.utils import plot_model


def getLog(fname: str, skip_header: bool) -> list:
    lines = []
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def loadImagesAndMeasurements(lines: list, base_img_dir: str, separator='/'):
    image_paths = []
    measurements = []
    for line in lines:
        # 0: center, left, right
        for i in range(3):
            current_path = base_img_dir + '/' + line[i].split(separator)[-1]
            image_paths.append(current_path)
        measurement = float(line[3])
        measurements.extend(
            [measurement, measurement + 0.2, measurement - 0.2])
    return (image_paths, measurements)


def generator(samples, batch_size=32, separator='/'):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for image_path, measurement in batch_samples:
                original_image = cv2.imread(image_path)
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                images.append(image)
                measurements.append(measurement)
                images.append(cv2.flip(image, 1))
                measurements.append(measurement * -1.0)
            yield sklearn.utils.shuffle(
                np.array(images), np.array(measurements))


def makeExampleImage(image_paths, measurements):
    images = []
    flipped_images = []
    image_lobels = ['center', 'left', 'right']
    flipped_image_lobels = ['flipped_center', 'flipped_left', 'flipped_right']
    for i in range(3):
        images.append(cv2.imread(image_paths[i]))
        flipped_images.append(cv2.flip(images[-1], 1))
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    i = 0
    for image in images:
        axes[0, i].imshow(image)
        axes[0, i].set_title(image_lobels[i])
        axes[0, i].axis('off')
        i += 1
    i = 0
    for image in flipped_images:
        axes[1, i].imshow(image)
        axes[1, i].set_title(flipped_image_lobels[i])
        axes[1, i].axis('off')
        i += 1
    plt.savefig("./output_images/training_data.png")


def makeModel():

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


# main
if __name__ == '__main__':
    BASE_DIR = './data'
    RAW_DATA = getLog(BASE_DIR + '/driving_log.csv', False)

    IMAGES, MEASUREMENTS = loadImagesAndMeasurements(RAW_DATA,
                                                     BASE_DIR + '/IMG', '\\')
    makeExampleImage(IMAGES, MEASUREMENTS)
    DATA = list(zip(IMAGES, MEASUREMENTS))

    TRAIN_SAMPLES, VALIDATION_SAMPLES = train_test_split(DATA, test_size=0.2)
    TRAIN_GENERATOR = generator(TRAIN_SAMPLES)
    VALIDATION_GENERATOR = generator(VALIDATION_SAMPLES)

    MODEL = makeModel()
    plot_model(
        MODEL, to_file='output_images/final_model.png', show_shapes=True)
    MODEL.compile(loss='mse', optimizer='adam')

    BATCH_SIZE = 32
    HISTORY_OBJECT = MODEL.fit_generator(
        TRAIN_GENERATOR,
        samples_per_epoch=math.ceil(len(TRAIN_SAMPLES) / BATCH_SIZE),
        validation_data=VALIDATION_GENERATOR,
        validation_steps=math.ceil(len(VALIDATION_SAMPLES) / BATCH_SIZE),
        epochs=1,
        verbose=1)

    plt.plot(HISTORY_OBJECT.history['loss'])
    plt.plot(HISTORY_OBJECT.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('output_images/loss.png')

    MODEL.save('model.h5')
