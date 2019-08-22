# Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense 
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# Traffic dataset loader
from load_signs_and_labels import loadTrafficSigns

train_rootpath = 'D:/TrafficSignRecognitionCNN/FinalDataSet/Training'
test_rootpath = 'D:/TrafficSignRecognitionCNN/FinalDataSet/Test'

train_samples = 37338       
test_samples = 2529
epochs = 10
batch_size = 64
img_width = 128
img_height = 128

#(train_images, train_class_labels, train_type_labels) = loadTrafficSigns(train_rootpath)
#(test_images, test_class_labels, test_type_labels) = loadTrafficSigns(test_rootpath)

# Each image is mapped to a single label from this array. [0-74]
sign_class = [
    "Uneven Road - Sag","Uneven Road - Bump","Slippery Road","Bend To Left","Bend To Right","Double Bend First To Left","Double Bend First To Right",
    "Pedestrians On Road","Cycle Route ","Cattle","Roadworks","Semaphore","Level Crossing With Gate or Barrier","Other Danger","Road Narrows On Both Sides",
    "Road Narrows On Left Side","Road Narrows On Right Side","Crossroads","Crossroads With Same Importance","Yield - Give Way","Give Way To Vehicles From Opposite Direction",
    "Stop","No Entry For Vehicular Traffic","No Cycling","No Vehicles Over Maximum Gross Weight Shown","No Truck","No Vehicles Over 2.10m Width ","No Vehicles Over 2.5m Height ",
    "No Vehicles Except Bicycles Being Pushed","No Left Turn","No Right Turn","No Overtaking","Maximum Speed 10km/h","Maximum Speed 20km/h","Maximum Speed 25km/h",
    "Maximum Speed 30km/h","Maximum Speed 40km/h","Maximum Speed 50km/h","Maximum Speed 60km/h","Maximum Speed 70km/h","Maximum Speed 80km/h","Maximum Speed 100km/h",
    "Maximum Speed 120km/h","Segregated Pedal Cycle And Pedestrian Route","Ahead Only","Turn Left","Turn Right","Ahead And Right Only","Roundabout","Cycles Only Route",
    "No Waiting","No Stopping","No Waiting 1-15","No Waiting 16-30","Ahead And Right Only","Keep Left","Keep Right","Car Parking Place","Truck Parking Place",
    "Bus Parking Place","Paralel Parking Only","Slow Traffic Zone","End Of Slow Traffic Zone","Go Straight","No Through Road For Vehicles","Roadworks Not Allowed",
    "Pedestrian Crossing","Bicycle Crossing","Parking Place On Right","Speed Bump","Right Of Way","Right Of Way End","Priority Over Oncoming Vehicles","Parking Place",
    "Handicapped Parking Place"]

sign_type = [
    "Warning sign",     # 0-19      W - 0
    "Order sign",       # 20-56     O - 1
    "Information sign"  # 57-74     I - 2
]

# Making the right input_shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Initialize the CNN
classifier = Sequential()

# Apply 3x3 convolution with 64 output filters and a 128x128 RGB image
# (2,2) will halve the input in both spatial dimenson
classifier.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# Flattens the input. Does not affect the batch size
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Rescale images [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_rootpath,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_rootpath,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary')

classifier.fit_generator(
    train_generator,
    steps_per_epoch=(train_samples//batch_size),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=(test_samples//batch_size))

classifier.save_weights('first_try.h5')