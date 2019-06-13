import os
import pandas

import numpy as np


from collections import namedtuple
from utils import BASE_DIR

data_dir = os.path.join(BASE_DIR, 'data', 'data_with_windows')
motion_filename = 'motion_data_22_users_with_window.csv'
stretch_filename = 'stretch_data_22_users_with_window.csv'
motion_filepath = os.path.join(data_dir, motion_filename)
stretch_filepath = os.path.join(data_dir, stretch_filename)


HarMotion = namedtuple('HarMotion', \
    'time user scenario trial window ax ay az gx gy gz label')
HarStretch = namedtuple('HarStretch', \
    'time user scenario trial window stretch label')

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

if __name__ == "__main__":
    read_csv_file(motion_filepath)
    pass
