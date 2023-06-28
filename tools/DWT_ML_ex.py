import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn import svm
import seaborn as sns
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

# Make array of all 12 calculated eatures
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
    return time_list, data_arr

time, train_data_pos = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/positive')
time, train_data_neg = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/negative')
time, test_data_pos = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/positive_test')
time, test_data_neg = load_data('/Users/josephaccurso/REU_git_repo/FPL_Datasets/datasets/negative_test')

# Get array of shape (#of training files) x ((# of phases) * (# of DWT decomps) * (# of calcualted feature stats))
def get_data_features(dataset, waveletname, level_val):
    index = 0
    num_features = 3 * (level_val + 1) * 12
    features = np.zeros(((len(dataset)), num_features))
    for file in range(len(dataset)):
        for phase in range(3):
            list_coeff = pywt.wavedec(dataset[file,:,phase], waveletname, level=level_val)
            for coeff_level in list_coeff:
                tmp = get_features(coeff_level)
                for i in range(len(get_features(coeff_level))):
                    features[file, index] = tmp[i]
                    index += 1
        index = 0
    list_features = np.asarray(features)
    return list_features

# Concatenate negative and positive sets to make one training set and one testing set
train_pos = get_data_features(train_data_pos, 'db2', 3)
train_neg = get_data_features(train_data_neg, 'db2', 3)
train = np.concatenate((train_pos, train_neg))

test_pos = get_data_features(test_data_pos, 'db2', 3)
test_neg = get_data_features(test_data_neg, 'db2', 3)
test = np.concatenate((test_pos, test_neg))

# Make labeled target columns for training set and testing set
train_labels_pos = np.ones((8630,1))
train_labels_neg = np.zeros((8630,1))
train_labels = np.concatenate((train_labels_pos, train_labels_neg))

test_labels_pos = np.ones((10,1))
test_labels_neg = np.zeros((10,1))
test_labels = np.concatenate((test_labels_pos, test_labels_neg))

# Perform random forest classification
clr = RandomForestClassifier(n_estimators=2000)
clr.fit(train, train_labels.ravel())
train_score = clr.score(train, train_labels.ravel())
test_score = clr.score(test, test_labels.ravel())
# Mean accuracy of training set (should be 1.0)
print(f'Train score for the dataset is about: {train_score}')
# Mean accuracy of testing set (1.0 is best)
print(f'Test Score for the dataset is about: {test_score}')
pred = clr.predict(test)
# Mean absolute error of testing set (lower = better)
mae = mean_absolute_error(test_labels, pred)
print(f'The mean absolute error for the dataset is: {mae}\n')
# Display confusion matrix
cm = confusion_matrix(test_labels, pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Perform gradient boost classification
cls = GradientBoostingClassifier(n_estimators=2000)
cls.fit(train, train_labels.ravel())
train_score = cls.score(train, train_labels.ravel())
test_score = cls.score(test, test_labels.ravel())
print(f'Train score for the dataset is about: {train_score}')
print(f'Test Score for the dataset is about: {test_score}')
pred = cls.predict(test)
mae = mean_absolute_error(test_labels, pred)
print(f'The mean absolute error for the dataset is: {mae}\n')
cm = confusion_matrix(test_labels, pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Perform support vector classification
clf = svm.SVC()
clf.fit(train, train_labels.ravel())
train_score = clf.score(train, train_labels.ravel())
test_score = clf.score(test, test_labels.ravel())
print(f'Train score for the dataset is about: {train_score}')
print(f'Test Score for the dataset is about: {test_score}')
pred = clf.predict(test)
mae = mean_absolute_error(test_labels, pred)
print(f'The mean absolute error for the dataset is: {mae}\n')
cm = confusion_matrix(test_labels, pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()