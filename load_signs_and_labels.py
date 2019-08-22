import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path


def loadTrafficSigns(roothpath):
    images = []                 # Images
    class_labels = []                 # Corresponding labels
    type_labels = []

    root_folder = Path(str(roothpath))              # receives folder 'D:/.../Training' or 'D:/.../Test'

    for sign_type in root_folder.iterdir():         # going into sign type folder (etc. INFORMATION_SIGNS)
        
        for sign_class in sign_type.iterdir():      # going into one of sign classes in INFORMATION_SIGNS folder

            prefix_path = str(roothpath) + '/' + str(sign_type.name) + '/' + str(sign_class.name) + '/'
            # D:/.../Training/A_WARRNING_SIGN_(0-19)/00000/ or D:/.../Test/A_WARRNING_SIGN_(0-19)/00000/
            idFile = open(str(prefix_path + 'ID_' + str(sign_class.name) + '.csv'))
            idReader = csv.reader(idFile, delimiter=';')

            for row in idReader:
                images.append(plt.imread(prefix_path + row[0]))
                class_labels.append(int(row[1]))
                type_labels.append(int(row[2]))

            print('Loaded from sign class: ' + sign_class.name)
    
    return images, class_labels, type_labels