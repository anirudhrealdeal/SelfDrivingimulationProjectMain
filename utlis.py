# Holds all functions
import random
import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.python.keras.optimizer_v1 import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2.cv2 as cv

tensorflow.compat.v1.disable_eager_execution()

def getName(filePath):
    return filePath.split('\\')[-1] # To get the last element ie names of images. everything before the \
#  '1+2+3'.split('+')[-1]
# Output: '3'

def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # print(data.head()) # Shows initial info
    # print(data['Center'][0], "\t", data['Steering'][0])
    # print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName) # apply function will automatically apply it to all images we have
    # print(data.head())
    print('Total images imported:', data.shape[0])
    return data

def balanceData(data, display=True):
    nBins = 31 # has to be odd number so that zero can be at center and we have +ve side and negative side
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBins)
    # print(bins)
    if display:
        center = (bins[:-1]+ bins[1:])*0.5 # Since we added them it almos becomes double range
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin,samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i]>= bins[j] and data['Steering'][i]<= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images:',len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining Images", len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin,samplesPerBin))
        plt.show()

    return data


def loadData(path,data):
    imagesPath =[]
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i] # Grabbing one entry of the data
        # print(indexedData)
        imagesPath.append(os.path.join(path,"IMG",indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augmentImage(imgPath, steering):
    # The reason we need steering is we can also use flip
# Left curve images can be flipped to right
    img = mpimg.imread(imgPath)
    # for pan
    if np.random.rand()<0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    # for zoom
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    # for Brightness
    if np.random.rand() < 0.5:
        brightness = iaa.MultiplyBrightness((0.6,1.2))
        img = brightness.augment_image(img)

    # for flip
    if np.random.rand() < 0.5:
        img = cv.flip(img, 1)
        steering = -steering

    return img, steering
# imgRe,st = augmentImage('test.jpg', 0)
# plt.imshow(imgRe)
# plt.show()

def preProcessing(img):
    # Steps for preprocessing
    # 1. Cropping to just the goal. ie the road
    img = img[60:135,:,:]
    # 2. change the colorspace
    img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    # 3. Add a blur
    img = cv.GaussianBlur(img, (3,3), 0)
    # 4. resize the imge to size of 200 x 66
    img = cv.resize(img, (200,66))
    # 5. Normalization
    img = img/255
    return img
# imgRe = preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgRe)
# plt.show()

def batchGenerator(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1) # Generates a random index
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield(np.asarray(imgBatch),np.asarray(steeringBatch))

def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2,2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2,2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(lr=0.0001), loss='mse')
    return model




