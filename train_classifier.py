from classifier import CarClassifier
from features_extractor import *
from processor import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from glob import glob
import os
import numpy as np


def shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def get_paths(subdirs, is_vehicles=True):

    if is_vehicles:
        type = 'vehicles'
    else:
        type = 'non-vehicles'

    root_path = os.path.join('.','training_data', type)
    pic_paths = []
    for dir in subdirs:
        path = os.path.join(root_path, dir, '*.png')
        list = glob(path)
        pic_paths.append(list)

    # Flatten the list
    picture_list = np.concatenate(pic_paths)
    return np.array(picture_list)


def read_data():

    # Vehicles
    subdirs = ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']
    vehicles = get_paths(subdirs, is_vehicles=True)
    vehicle_labels = np.ones(len(vehicles))

    subdirs = ['Extras', 'GTI']
    non_vehicles = get_paths(subdirs, is_vehicles=False)
    non_vehicle_labels = np.zeros(len(non_vehicles))

    image_paths = np.concatenate((vehicles, non_vehicles))
    labels = np.concatenate((vehicle_labels, non_vehicle_labels))

    return np.array(image_paths), np.array(labels)


def main():

    print('Loading data.')
    image_paths, y = read_data()
    print('{} Images found.'.format(len(image_paths)))

    hist = np.histogram(y,2,)

    print('Extracting features.')
    # feature_extractor = VehicleFeatureExtractor()
    feature_extractor = VehicleFeatureExtractor(color_space='YCrCb',# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                                                orient=9,  # HOG orientations
                                                pix_per_cell=8,  # HOG pixels per cell
                                                cell_per_block=2,  # HOG cells per block
                                                hog_channel=0,  # Can be 0, 1, 2, or "ALL"
                                                spatial_size=(32, 32),  # Spatial binning dimensions
                                                hist_bins=32,  # Number of histogram bins
                                                spatial_feat=True,  # Spatial features on or off
                                                hist_feat=True,  # Histogram features on or off
                                                hog_feat=True  # HOG features on or off
                                                )

    X = feature_extractor.extract_features(image_paths)

    print('Training classifier')
    car_classifier = CarClassifier()
    acc = car_classifier.fit(X, y)

    # Save the current state
    print('Saving model.')
    car_classifier.save_state('car_classifier_model.pkl')

    feature_extractor.save('feature_extractor_settings.pkl')

    print('Model accuracy: {}'.format(acc))

if __name__ == "__main__":
    main()