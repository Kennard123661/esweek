import os
import pandas

import numpy as np


from collections import namedtuple
from utils import BASE_DIR
from scipy.io import loadmat

data_dir = os.path.join(BASE_DIR, 'data', 'data_with_windows')
motion_filename = 'motion_data_22_users_with_window.csv'
stretch_filename = 'stretch_data_22_users_with_window.csv'
motion_filepath = os.path.join(data_dir, motion_filename)
stretch_filepath = os.path.join(data_dir, stretch_filename)

feature_filepath = os.path.join(BASE_DIR, 'data', 'baseline_classifier', 'features_file.mat')

HarMotion = namedtuple('HarMotion', \
    'time user scenario trial window ax ay az gx gy gz label')
HarStretch = namedtuple('HarStretch', \
    'time user scenario trial window stretch label')

NUM_FEATURES_TO_NORMALIZE = 119
NUM_INPUT_FEATURES = 120

UNDEFINED_LABEL = 10

def read_csv_file(filepath):
    """ Reads csv file """
    assert filepath.split('.')[-1] == 'csv'
    data = pandas.read_csv(filepath).reset_index()
    return np.array(data.values)

def get_data():
    """ Returns Human Activity Recognition Data (HAR) as motion and 
    stretch data """
    har_motions = read_csv_file(motion_filepath)
    har_motions = np.array([
        HarMotion(*har_motion) for har_motion in har_motions])
    har_stretchs = read_csv_file(stretch_filepath)
    har_stretchs = np.array([
        HarStretch(*har_stretch) for har_stretch in har_stretchs])
    return har_motions, har_stretchs

def _get_data_and_labels(features, labels, idxs):
    return features[idxs], labels[idxs]

def remove_undefined_data(data, labels):
    undefined_data == 10

def get_feature_data():
    """ Returns features """
    feature_data = loadmat(feature_filepath)
    normed_features = feature_data['feature_matrix_norm']
    feature_mean = feature_data['feature_mean_vector']
    feature_variance = feature_data['feature_variance_vector']
    labels = feature_data['label_vector']
    
    train_idxs = np.reshape(feature_data['train_idx'], [-1]) - 1
    val_idxs = np.reshape(feature_data['xval_idx'], [-1]) - 1
    test_idxs = np.reshape(feature_data['test_idx'], [-1]) - 1

    features = normed_features[:,:NUM_INPUT_FEATURES]
    train_data, train_labels = _get_data_and_labels(features, labels, train_idxs)
    val_data, val_labels = _get_data_and_labels(features, labels, val_idxs)
    test_data, test_labels = _get_data_and_labels(features, labels, test_idxs)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    

if __name__ == "__main__":
    get_feature_data()    
    pass
