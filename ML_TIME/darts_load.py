import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # gets rid of harmless error messages
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
os.environ['NCCL_DEBUG'] = 'WARN'
import pickle
import numpy as np
import pandas as pd 
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score

from keras import backend as K

from keras import Model, Sequential
from keras.layers import (Input, Reshape, Rescaling, 
                                    TimeDistributed, MaxPool1D, BatchNormalization, 
                                    Embedding, Dense, Dropout,
                                    Flatten, Softmax)

from search_layers import (regularizer, kernel_init, 
                            Zero, MultiHeadEncoderAttention, MultiHeadDecoderAttention,
                            Densely, Conv, SepConv, DilConv, Identity)


from keras.utils import plot_model
from keras.models import load_model
import imageio

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef

from keras.callbacks import ModelCheckpoint


import gc
from collections import namedtuple

import sys
from graphviz import Digraph

from plot_utils import train_curves, plot

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc

from darts_transformer_temp import OPS

# from darts_search_arch_genotype_v2 import darts_search_9, darts_search_10

Genotype = namedtuple('Genotype', ['normal', 'normal_concat'])


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    # strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

X = np.load("assets/vanilla_X_norm.npy", mmap_mode="r")
y = np.load("assets/vanilla_y_norm.npy", mmap_mode="r")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
print(X_tr.shape, y_tr.shape)
print(X_te.shape, y_te.shape)


cfg = {
    # general setting
    "batch_size": 121,
    "init_channels": 64,
    "layers": 4,
    "num_typ_classes": 46,
    # "num_loc_classes": 15,
    "sub_name": 'darts_search',

    # training setting
    "epoch": 10,
    "start_search_epoch": 15,
    "init_lr": 0.001,
    "momentum": 0.9,
    "weights_decay": 3e-4,
    "grad_clip": 10.0,

    "arch_learning_rate": 0.001,
    "arch_weight_decay": 0.001,
    }


tr_shape = X_tr.shape[0]
X_train1 = tf.convert_to_tensor(X_tr[:(tr_shape//8)*1])
X_train2 = tf.convert_to_tensor(X_tr[(tr_shape//8)*1: (tr_shape//8)*2])
X_train3 = tf.convert_to_tensor(X_tr[(tr_shape//8)*2: (tr_shape//8)*3])
X_train4 = tf.convert_to_tensor(X_tr[(tr_shape//8)*3: (tr_shape//8)*4])
X_train5 = tf.convert_to_tensor(X_tr[(tr_shape//8)*4: (tr_shape//8)*5])
X_train6 = tf.convert_to_tensor(X_tr[(tr_shape//8)*5: (tr_shape//8)*6])
X_train7 = tf.convert_to_tensor(X_tr[(tr_shape//8)*6: (tr_shape//8)*7])
X_train8 = tf.convert_to_tensor(X_tr[(tr_shape//8)*7:])
X_train = tf.concat([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8], axis=0)

te_shape = X_te.shape[0]
X_test1 = tf.convert_to_tensor(X_te[: (te_shape//4)*1])
X_test2 = tf.convert_to_tensor(X_te[(te_shape//4)*1:(te_shape//4)*2])
X_test3 = tf.convert_to_tensor(X_te[(te_shape//4)*2:(te_shape//4)*3])
X_test4 = tf.convert_to_tensor(X_te[(te_shape//4)*3:])
X_test = tf.concat([X_test1, X_test2, X_test3, X_test4], axis=0)

y_train = tf.convert_to_tensor(y_tr)
y_test = tf.convert_to_tensor(y_te)

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,:20])) #  y_train[:,20:]))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,:20])) #  y_test[:,20:]))
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,:46]))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,:46]))

train_dataset = train_dataset.shuffle(4000).batch(cfg["batch_size"], drop_remainder=True)#.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(cfg["batch_size"], drop_remainder=True)#.prefetch(tf.data.experimental.AUTOTUNE)

with open('assets/pc_darts_search_history_v2', "rb") as file_pi:    # path to load model history
    history = pickle.load(file_pi)



plt.rcParams.update({'legend.fontsize': 12,
                    'axes.labelsize': 16, 
                    'axes.titlesize': 16,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16})


loaded_transformer_model = load_model('assets/darts_transformer_model', compile=False)   # path of complete model
loaded_transformer_model.summary()
layers = loaded_transformer_model.layers
first_layer = layers[0]
print(first_layer.input_shape)
for l in loaded_transformer_model.layers:
    print(l.input_shape)



loaded_transformer_model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], 
                optimizer = Adam(learning_rate=0.001),
                metrics={"type":[ 
                                CategoricalAccuracy(name="acc"),
                                MatthewsCorrelationCoefficient(num_classes=46, name ="mcc"),
                                F1Score(num_classes=46, name='f1_score')
                                ] 
                        }
                )



# loaded_transformer_model.evaluate(test_dataset, verbose=1)

type_names = ["exciting_Class1","exciting_Class2","exciting_Class3","exciting_Class4","exciting_Class5", "exciting_Class6","exciting_Class7","exciting_Class8","exciting_Class9","exciting_Class10", "exciting_Class11","exciting_tt","exciting_ww",
              'series_Class1','series_Class2','series_Class3','series_Class4','series_Class5','series_Class6','series_Class7','series_Class8','series_Class9','series_Class10','series_Class11','series_tt','series_ww',
               'transformer_Class1','transformer_Class2','transformer_Class3','transformer_Class4','transformer_Class5','transformer_Class6','transformer_Class7','transformer_Class8','transformer_Class9','transformer_Class10','transformer_Class11','transformer_tt','transformer_ww',
                "Capacitor_Switch", "external_fault","ferroresonance",  "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"]

# plt.rcParams.update({'legend.fontsize': 14,
#                     'axes.labelsize': 18, 
#                     'axes.titlesize': 18,
#                     'xtick.labelsize': 18,
#                     'ytick.labelsize': 18})


# print(darts_search_9.normal)
# print(darts_search_10[0])
# print(len(darts_search_10[0]))

with open("ML_TIME/darts_search_arch_genotype_v2.py") as graph_file:
    graphs = graph_file.readlines()
    epoch = 0
    for i, g in enumerate(graphs):
        if i%2 != 0:
            genotype = eval(g.split(" = ")[1])
            
            plot(list(genotype.normal), "assets/pc_darts_genotypes/normal", str(epoch+1))

            epoch+=1
        # else skips blank line


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

filenames = ["assets/pc_darts_genotypes/" + f for f in sorted_alphanumeric(os.listdir("assets/pc_darts_genotypes")) if f.endswith(".png")]


images = []
for filename in filenames:
    images.append(imageio.imread(filename))


