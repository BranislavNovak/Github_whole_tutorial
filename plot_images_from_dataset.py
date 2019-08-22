import matplotlib.pyplot as plt
import random
import numpy as np

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
    "Warning sign",     # 0-19      W
    "Order sign",       # 20-56     O
    "Information sign"  # 57-74     I
]

def ploting(train_images, train_class_labels, train_type_labels, test_images, test_class_labels, test_type_labels):
    
    fig = plt.figure(figsize=(20,20))
    columns = 8
    rows = 8

    for i in range(1, columns*rows + 1):
        elem = random.randint(0,len(train_images)-1)
        img = train_images[elem]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.ylabel(str(train_type_labels[elem]))
        plt.xlabel(str(train_class_labels[elem]))
        
    plt.show()

    fig = plt.figure(figsize=(20,20))
    columns = 8
    rows = 8

    for i in range(1, columns*rows + 1):
        elem = random.randint(0,len(test_images)-1)
        img = test_images[elem]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.ylabel(str(test_type_labels[elem]))
        plt.xlabel(str(test_class_labels[elem]))
        
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap='binary')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(sign_type[predicted_label], 100*np.max(predictions_array), sign_type[true_label], color=color))


def plot_value_array(i, predictions_array, true_label):
  prediction_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), prediction_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(prediction_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')