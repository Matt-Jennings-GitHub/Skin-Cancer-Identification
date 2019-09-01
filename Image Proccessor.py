# Modules
from PIL import Image
import cv2
import os

#Variables
rows, cols = 64, 64

# Paths
input_path = 'Training Data/MalignantTestSource'
output_path = 'Training Data/MalignantTestProcessed'

for img_name in os.listdir(input_path):
    # Input Image
    print(img_name)
    img = cv2.imread('{}/{}'.format(input_path, img_name)) # cv2 import as BGR np array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

    # Crop outer edges
    w, h = img.shape[1], img.shape[0]
    crop = 0.15
    img = img[int(h*crop):h-int(h*crop), int(w*crop):w-int(w*crop)]

    # Crop to square
    w, h = img.shape[1], img.shape[0]
    if w > h : # Crop to square
        img = img[0:h, int((w-h)/2):w-int((w-h)/2)]
    elif h > w :
        img = img[int((h - w) / 2):h - int((h - w) / 2), 0:w]
    # Rescale to target size
    img = cv2.resize(img, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)

    # Display
    img = Image.fromarray(img, 'RGB')
    img.save('{}/{}.png'.format(output_path,img_name.split('.')[0]), 'PNG')


