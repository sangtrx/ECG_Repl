import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold

STEP = 4864

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def import_key_data(path):
    labels=[]
    ecg_filenames=[]
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                labels.append(header_data[15][5:-1])
                ecg_filenames.append(filepath)
    return labels, ecg_filenames


def make_undefined_class(labels, df_unscored):
    df_labels = pd.DataFrame(labels)
    for i in range(len(df_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(df_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    return df_labels

def onehot_encode(df_labels):
    one_hot = MultiLabelBinarizer()
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    print("The classes we will look at are encoded as SNOMED CT codes:")
    print(one_hot.classes_)
    y = np.delete(y, -1, axis=1)
    print("classes: {}".format(y.shape[1]))
    return y, one_hot.classes_[0:-1]

def get_labels_for_all_combinations(y):
    y_all_combinations = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_all_combinations

def split_data(labels, y_all_combo):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(labels,y_all_combo))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds

def split_data_opt(labels, y_all_combo):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(labels,y_all_combo))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds
        
def shuffle_batch_generator(batch_size, gen_x, gen_y, order_array): 
    # np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size, STEP, 12))
    batch_labels = np.zeros((batch_size, 19, 27)) #drop undef class
    while True:
        for i in range(batch_size):
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            
        yield batch_features, batch_labels

def generate_y_shuffle(y_train, order_array, ecg_lengths):
    while True:
        for i in order_array:
            trunc = int(ecg_lengths[i]/256)
            if (trunc >= 19):
                y_shuffled = np.ones(shape=(19,27))*y_train[i]
            else: 
                y_shuffled = np.concatenate((np.ones(shape=(trunc, 27))*y_train[i], np.zeros(shape=(19-trunc,27))), axis=0)
            yield y_shuffled

def generate_X_shuffle(X_train, order_array):
    while True:
        for i in order_array:
            #if filepath.endswith(".mat"):
            data, header_data = load_challenge_data(X_train[i])
            X_train_new = pad_sequences(data, maxlen=STEP, truncating='post',padding="post")
            X_train_new = X_train_new.reshape(STEP,12)
            yield X_train_new

def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

def generate_validation_data(ecg_filenames, y,test_order_array, ecg_lengths):

    ecg_train_timeseries=[]
    y_train_timeseries=[]
    for i in test_order_array:
        data, header_data = load_challenge_data(ecg_filenames[i])
        data = pad_sequences(data, maxlen=STEP, truncating='post',padding="post")
        ecg_train_timeseries.append(data)

        trunc = int(ecg_lengths[i]/256)
        if (trunc >= 19):
            y_shuffled = np.ones(shape=(19,27))*y[i]
        else: 
            y_shuffled = np.concatenate((np.ones(shape=(trunc, 27))*y[i], np.zeros(shape=(19-trunc,27))), axis=0)
        y_train_timeseries.append(y_shuffled)
    X_train_gridsearch = np.transpose(np.asarray(ecg_train_timeseries), axes=(0,2,1))
    y_train_gridsearch = np.asarray(y_train_timeseries)

    return X_train_gridsearch, y_train_gridsearch

def get_signal_length(data_path):
    signal_length=[]
    for subdir, dirs, files in sorted(os.walk(data_path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                splitted = header_data[0].split()
                signal_length.append(splitted[3])
    
    return signal_length














