# Modules
import os
from PIL import Image
import cv2
from pathlib import Path
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model, model_from_json
from keras.callbacks import TensorBoard

# Variables
benign_path = 'Training Data/Benign'
malignant_path = 'Training Data/Malignant'
target_size = (64, 64)
crop_percentage = 0.1
train_fraction = 0.8

epochs = 10
batch_size = 32

# Preprocess Images
def preprocess_image(input_path, target_size, crop_percentage): # Returns np array of processed image
    # Input Image
    img = cv2.imread(input_path) # cv2 import as BGR np array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

    # Crop outer edges
    w, h = img.shape[1], img.shape[0]
    img = img[int(h*crop_percentage):h-int(h*crop_percentage), int(w*crop_percentage):w-int(w*crop_percentage)]

    # Crop to square
    w, h = img.shape[1], img.shape[0]
    if w > h : # Crop to square
        img = img[0:h, int((w-h)/2):w-int((w-h)/2)]
    elif h > w :
        img = img[int((h - w) / 2):h - int((h - w) / 2), 0:w]

    # Rescale to target size
    img = cv2.resize(img, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)

    return img

# Process images
def process_images():
    for path in (benign_path, malignant_path):
        # Clear existing processed images
        for img_name in os.listdir(path+'Processed'):
            os.remove(path+'Processed/{}'.format(img_name))
        # Preprocess images
        for img_name in os.listdir(path+'Source'):
            print(img_name)
            img = preprocess_image('{}/{}'.format(path+'Source', img_name), target_size, crop_percentage)

            # Save processed Images
            img = Image.fromarray(img, 'RGB')
            img.save('{}/{}.png'.format(path+'Processed', img_name.split('.')[0]), 'PNG')
    return

reprocess = False
if reprocess:
    process_images()

# Load processed images
images = []
labels = []
for path in (benign_path, malignant_path):
    for img_name in os.listdir(path+'Processed'):
        img = cv2.imread(path+'Processed/{}'.format(img_name))

        # Form input and label arrays
        images.append(img)
        if 'Benign' in path:
            labels.append(0)
        if 'Malignant' in path:
            labels.append(1)

x_train, x_test = np.array(images[0:int(len(images)*train_fraction)], dtype=np.float32), np.array(images[int(len(images)*train_fraction):-1], dtype=np.float32) # Train Test split
y_train, y_test = np.array(labels[0:int(len(labels)*train_fraction)], dtype=np.float32), np.array(labels[int(len(labels)*train_fraction):-1], dtype=np.float32)
x_train /= 255
x_test /= 255

# Define Network
logger = TensorBoard(log_dir='logs', write_graph=True)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(target_size[0], target_size[1], 3), activation='relu', name='Convolutional_Layer_1'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='Convolutional_Layer_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='Convolutional_Layer_3'))
model.add(Conv2D(64, (3, 3), activation='relu', name='Convolutional_Layer_4'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu', name='Final_Hidden_Layer'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name='Output_Layer'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train Network
def train_network(epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, callbacks=[logger]) #metrics=['accuracy']
    results = model.evaluate(x_test, y_test, verbose=0)
    print('Validation Loss: {} Validation Accuracy: {}'.format(results[0], results[1]))

    # Save Model
    model_structure = model.to_json()
    f = Path("model_structure.json")
    f.write_text(model_structure)
    model.save_weights("model_weights.h5")

    return

retrain = True
if retrain:
    train_network(epochs, batch_size)

# Load Model
f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

# Test Image
def test_image(test_path):
    img = preprocess_image(test_path, target_size, crop_percentage)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)[0][0] # Predict
    print("Malignance Confidence: {:.4f}%".format(int(result * 100)))
    return

for path in ('Training Data/Tests/Benign', 'Training Data/Tests/Malignant'):
    print(path)
    for img_name in os.listdir(path):
        print('{}/{}'.format(path, img_name))
        test_image('{}/{}'.format(path,img_name))


















