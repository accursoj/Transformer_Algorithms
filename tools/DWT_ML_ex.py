import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.ensemble import GradientBoostingClassifier
import scipy
import pandas as pd

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    tmp = []
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    tmp = [entropy] + crossings + statistics
    features_list = np.asarray(tmp)
    return features_list

def load_data(directory):
    file_list = []
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            file_list.append(f)

    raw_data = np.asarray(pd.read_csv(f, header=None))
    data_arr = np.zeros((len(file_list), len(raw_data), 3))
    for i in range(len(file_list)):
        raw_data = np.asarray(pd.read_csv(file_list[i], header=None))
        time_list = []
        for j in range(len(raw_data)):
            time_list.append(raw_data[j,0])
            for k in range(3):
                data_arr[i,j,k] = raw_data[j,k+1]
    #print(len(data_arr))
    # print(len(time_list))
    return time_list, data_arr

time, train_data_pos = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/positive')
time, train_data_neg = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/negative')
time, test_data_pos = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/positive_test')
time, test_data_neg = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/negative_test')


def get_data_features(dataset, waveletname, level_val):
    index = 0
    features = np.zeros(((len(dataset)), 3, (level_val + 1), 12))
    for file in range(len(dataset)):
        for phase in range(3):
            list_coeff = pywt.wavedec(dataset[file,:,phase], waveletname, level=level_val)
            for coeff_level in list_coeff:
                tmp = get_features(coeff_level)
                index += index
                for i in range(12):
                    features[file, phase, index, i] = tmp[i]
    list_features = np.asarray(features)
    return list_features

train_pos = get_data_features(train_data_pos, 'db1', 1)
train_neg = get_data_features(train_data_neg, 'db1', 1)
test_pos = get_data_features(test_data_pos, 'db1', 1)
test_neg = get_data_features(test_data_neg, 'db1', 1)



