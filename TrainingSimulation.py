# Link for the simulator. I downloaded the windows version
# https://github.com/udacity/self-driving-car-sim
import matplotlib.pyplot as plt

print('Setting Up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # this is to not give any warnings
import socketio
import eventlet
from flask import Flask
# Step 1: Importing the data information
from utlis import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
path = 'myData'
data = importDataInfo(path)

# Step 2: Visualization and distribution of your data
data = balanceData(data, display=False)

# Step 3: Put all image values in one list and steering values in another.
# Data is rn in pandas format we have to convert to numpy
# We have to add em to a list and convert to numpy arrays
imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])

# Step 4: Splitting of the data into training and validation
x_train, x_validation, y_train, y_validation = train_test_split(imagesPath,steering, test_size=0.2, random_state=5)
# print('Total training images:', len(x_train))
# print('Total validation images:', len(x_validation))

# Step 5: Augmentation of data
# It's never enough with the data. We can zoom, pan the image, change lighting and create new data
# We have a library for that
# All augmentation will happen during the training process


# Step 6: Preprocessing. Why do we need to preprocess?
# 1.we want to crop. we dont want the car in the image and neither do we want the mountains etc. we just want the roads


# Step 7: Batch generator. We do not send all our images together to our training model. We send them in batches.
# This helps in generalization and gives freedom to how we can create images and send them to model
# Before sending to model we should augment and preprocess our image


# Step 8: Creating the model. We are  using NVIDIA's model
model = createModel()
model.summary()

# Step 9: Main Step. Training our model
history = model.fit(batchGenerator(x_train,y_train,100,1), steps_per_epoch=300,
          epochs=10, validation_data=batchGenerator(x_validation,y_validation,100,0), validation_steps=200)

# Step 10: save the model
model.save('model.h5')
print('Model is saved...')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Valiidation'])
plt.title('Loss')
plt.ylim([0,0.2])
plt.xlabel('Epoch')
plt.show()



