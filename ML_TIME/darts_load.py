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
                                    Flatten, Softmax, MultiHeadAttention, LayerNormalization, Add, Concatenate)

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

from plot_utils import plot
from plot_utils import train_curves as DARTS_train_curves

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves

import glob
from PIL import Image

# number of classes (including a no fault class)
NUM_CLASSES = 46
INPUT_SHAPE = (726, 3)
EPOCHS = 10

Genotype = namedtuple('Genotype', ['normal', 'normal_concat'])


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    # strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

# X = np.load("assets/vanilla_X_norm.npy", mmap_mode="r")
# y = np.load("assets/vanilla_y_norm.npy", mmap_mode="r")

# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
# print(X_tr.shape, y_tr.shape)
# print(X_te.shape, y_te.shape)


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

def TransformerEncoder(inputs, num_heads, head_size, dropout, units_dim):
    encode1 = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size , dropout=dropout
    )(encode1, encode1)
    encode2 = Add()([attention_output, encode1])
    encode3 = LayerNormalization(epsilon=1e-6)(encode2)
    for units in [units_dim * 2, units_dim]:
        encode3 = Dense(units=units, activation='relu')(encode3)
        encode3 = Dropout(dropout)(encode3)
    outputs = Add()([encode3, encode2])

    return outputs



OPS = {'none': lambda units, wd: 
           Zero(),
       'max_pool_2': lambda units, wd:
           MaxPool1D(2, strides = 1, padding='same'),
       'skip_connect': lambda units, wd:
           Identity(),
       'encoder_att': lambda units, wd:
           MultiHeadEncoderAttention(units, wd),
       'decoder_att': lambda units, wd:
           MultiHeadDecoderAttention(units, wd),
       'dense': lambda units, wd:
           Densely(units, wd),
       'sep_conv_1': lambda units, wd:
           SepConv(units, 1, wd),
       'sep_conv_3': lambda units, wd:
           SepConv(units, 3, wd),
       'sep_conv_5': lambda units, wd:
           SepConv(units, 5, wd),
       'conv_1': lambda units, wd:
           Conv(units, 1, wd),
       'conv_3': lambda units, wd:
           Conv(units, 3, wd),
       'conv_5': lambda units, wd:
           Conv(units, 5, wd),
       'dil_conv_1': lambda units, wd:
           DilConv(units, 1, 2, wd),
       'dil_conv_3': lambda units, wd:
           DilConv(units, 3, 2, wd),
       'dil_conv_5': lambda units, wd:
           DilConv(units, 5, 2, wd),
           }

class Cell(tf.keras.layers.Layer):
    """Cell Layer"""
    def __init__(self, genotype, ch, wd, name='Cell', **kwargs):
        super(Cell, self).__init__(name=name, **kwargs)

        self.wd = wd
        self.genotype = genotype
        
        self.preprocess0 = Densely(ch, wd=wd)
        self.preprocess1 = Densely(ch, wd=wd)

        self._ops = []
        for op_name, _ in self.genotype.normal:
            self._ops.append(OPS[op_name](ch, wd))
    
    def call(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for op in self._ops:
            s = 0
            for j, h in enumerate(states):
                branch = op(h, weights[offset + j])
                s += branch
            offset += len(states)
            states.append(s)
        return tf.concat(states[-self.genotype.normal_concat[-1]:], axis=-1)
    
class DARTS_Transformer(Model):
    def __init__(self, cfg, genotype, input_shape, num_classes):
        super(DARTS_Transformer, self).__init__()
        self.genotype = genotype
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.cfg = cfg

        self.cell = Cell(genotype, ch=self.cfg['init_channels'], wd=self.cfg['weights_decay'])

        self.model = self.build_transformer_model()

    def call(self, inputs):
        x = self.cell(inputs, inputs, weights=None)
        x = self.model(x)
        return x

    def build_transformer_model(self):
        input_sig = Input(shape=(726, 3))   # shape = shape of single data file
        sig = input_sig/6065.3965
        sig = Reshape((6, 121, 3))(sig)     # reshape data file (ex. (726, ...) --> (6, 121, ...))
        sig = TimeDistributed(Flatten())(sig)

        sig = Dense(1024, activation="relu")(sig)
        sig = Dropout(0.2)(sig)
        sig = Dense(64, activation="relu")(sig)
        sig = Dropout(0.2)(sig)

        embeddings = Embedding(input_dim=6, output_dim=64)  # input_dim = value from reshaped data: Reshape((input_dim, ..., ...))
        position_embed = embeddings(tf.range(start=0, limit=6, delta=1))    # limit = input_dim
        sig = sig + position_embed

        for e in range(4):
            sig = TransformerEncoder(sig, num_heads=4, head_size=64, dropout=0.2, units_dim=64)

        sig = Flatten()(sig)

        typ = Dense(256, activation="relu")(sig)
        typ = Dropout(0.2)(typ)
        typ = Dense(128, activation="relu")(typ)
        typ = Dense(32, activation="relu")(typ)
        typ = Dropout(0.2)(typ)
        typ_output = Dense(NUM_CLASSES, activation="softmax", name="type")(typ)


        # initalize model
        model = Model(inputs=input_sig, outputs=[typ_output])

        model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], 
                    optimizer = Adam(learning_rate=0.001),
                    metrics={"type":[ 
                                        CategoricalAccuracy(name="acc"),
                                        MatthewsCorrelationCoefficient(num_classes=NUM_CLASSES, name ="mcc"),
                                        F1Score(num_classes=NUM_CLASSES, name='f1_score')
                                    ] 
                                }
                        )

        model._name = "DARTS_Transformer_Model"

        return model

def prepare_tensors(norm=None):
    if norm:
        print('Loading normalized data...')
        X = np.load("assets/vanilla_X_norm.npy", mmap_mode="r")
        y = np.load("assets/vanilla_y_norm.npy", mmap_mode="r")

        # create 80:20 training-testing split of  data
        print('Splitting normalized data...')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
        print(X_tr.shape, y_tr.shape)
        print(X_te.shape, y_te.shape)

        print('Converting to tensors...')
        # convert numpy arrays to tensors
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
        X_test1 = tf.convert_to_tensor(X_te[:(te_shape//4)*1])
        X_test2 = tf.convert_to_tensor(X_te[(te_shape//4)*1:(te_shape//4)*2])
        X_test3 = tf.convert_to_tensor(X_te[(te_shape//4)*2:(te_shape//4)*3])
        X_test4 = tf.convert_to_tensor(X_te[(te_shape//4)*3:])
        X_test = tf.concat([X_test1, X_test2, X_test3, X_test4], axis=0)

        y_train = tf.convert_to_tensor(y_tr)
        y_test = tf.convert_to_tensor(y_te)

        print('Normalized tensors:')
        print(f'X_train, y_train shapes: {X_train.shape}, {y_train.shape}')
        print(f'X_test, y_test shapes: {X_test.shape}, {y_test.shape}')

    else:
        print('Loading standard data...')
        X = np.load("assets/signals_full.npy", mmap_mode="r")
        y = np.load("assets/signals_gts3_full.npy", mmap_mode="r")

        # create 80:20 training-testing split of  data
        print('Splitting standard data...')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
        print(X_tr.shape, y_tr.shape)
        print(X_te.shape, y_te.shape)

        # convert numpy arrays to tensors
        print('Converting to tensors...')
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
        X_test1 = tf.convert_to_tensor(X_te[:(te_shape//4)*1])
        X_test2 = tf.convert_to_tensor(X_te[(te_shape//4)*1:(te_shape//4)*2])
        X_test3 = tf.convert_to_tensor(X_te[(te_shape//4)*2:(te_shape//4)*3])
        X_test4 = tf.convert_to_tensor(X_te[(te_shape//4)*3:])
        X_test = tf.concat([X_test1, X_test2, X_test3, X_test4], axis=0)

        y_train = tf.convert_to_tensor(y_tr)
        y_test = tf.convert_to_tensor(y_te)

        print('Standard tensors:')
        print(f'X_train, y_train shapes: {X_train.shape}, {y_train.shape}')
        print(f'X_test, y_test shapes: {X_test.shape}, {y_test.shape}')

    return X_train, X_test, y_train, y_test







with open("ML_TIME/darts_search_arch_genotype_v2.py") as graph_file:
    graphs = graph_file.readlines()
    epoch = 0
    for i, g in enumerate(graphs):
        if i%2 != 0:
            genotype = eval(g.split(" = ")[1])
            
            plot(list(genotype.normal), "assets/pc_darts_genotypes/normal", str(epoch+1))

            epoch+=1
            final_genotype = genotype
        # else skips blank line


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

filenames = ["assets/pc_darts_genotypes/" + f for f in sorted_alphanumeric(os.listdir("assets/pc_darts_genotypes")) if f.endswith(".png")]

def make_gif(frame_folder):
    frames = [Image.open(image) for image in frame_folder]
    frame_one = frames[0]
    frame_one.save("/home/msayler/DARTS_gitRepo/Transformer_Algorithms/assets/genotypes.gif", format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=1)
make_gif(filenames)
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('assets/genotypes.gif', images)

with strategy.scope():
    DARTS_model = DARTS_Transformer(cfg=cfg, genotype=final_genotype, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

DARTS_model = DARTS_model.model
DARTS_model.summary()
plot_model(DARTS_model, to_file='assets/DARTS_model.png', expand_nested=True, show_shapes=True)

checkpoint_filepath = "assets/DARTS_checkpoint_weights.h5"    # path to save checkpoint weights
# # uncomment if you want to start the training on pre-existing checkpoint weights
# DARTS_model.load_weights(checkpoint_filepath)


# begin training with standard data
X_train, X_test, y_train, y_test = prepare_tensors(norm=False)
transformer_model_history = DARTS_model.fit(X_train,
                                                y_train,
                                                epochs = 1,
                                                batch_size = 64 * strategy.num_replicas_in_sync,
                                                validation_data = (X_test, y_test),   # validate against test data
                                                validation_batch_size = 64 * strategy.num_replicas_in_sync,
                                                verbose = 1,
                                                callbacks = [ModelCheckpoint(filepath = checkpoint_filepath,
                                                                                verbose = 1,
                                                                                monitor = "val_loss",
                                                                                save_best_only = True,
                                                                                save_weights_only = True,
                                                                                mode = "min")
                                                            ]
                                                )

# train model with normalized data
X_train, X_test, y_train, y_test = prepare_tensors(norm=True)
DARTS_model.load_weights(checkpoint_filepath)
transformer_model_history = DARTS_model.fit(X_train,
                                                y_train,
                                                epochs = EPOCHS,
                                                batch_size = 64 * strategy.num_replicas_in_sync,
                                                validation_data = (X_test, y_test),   # validate against test data
                                                validation_batch_size = 64 * strategy.num_replicas_in_sync,
                                                verbose = 1,
                                                callbacks = [ModelCheckpoint(filepath = checkpoint_filepath,
                                                                                verbose = 1,
                                                                                monitor = "val_loss",
                                                                                save_best_only = True,
                                                                                save_weights_only = True,
                                                                                mode = "min")
                                                            ]
                                                )


with open('assets/DARTS_model_fault_detr_history_full', 'wb') as file_pi:    # path to save model history
    pickle.dump(transformer_model_history.history, file_pi)

with open('assets/DARTS_model_fault_detr_history_full', "rb") as file_pi:    # path to load model history
    history = pickle.load(file_pi)

DARTS_model.load_weights(checkpoint_filepath)
DARTS_model.save('assets/DARTS_transformer_model.keras')  # path to save complete model


test_metrics = DARTS_model.evaluate(X_test, y_test)
test_metrics

type_names = ["exciting_Class1","exciting_Class2","exciting_Class3","exciting_Class4","exciting_Class5", "exciting_Class6","exciting_Class7","exciting_Class8","exciting_Class9","exciting_Class10", "exciting_Class11","exciting_tt","exciting_ww",
              'series_Class1','series_Class2','series_Class3','series_Class4','series_Class5','series_Class6','series_Class7','series_Class8','series_Class9','series_Class10','series_Class11','series_tt','series_ww',
               'transformer_Class1','transformer_Class2','transformer_Class3','transformer_Class4','transformer_Class5','transformer_Class6','transformer_Class7','transformer_Class8','transformer_Class9','transformer_Class10','transformer_Class11','transformer_tt','transformer_ww',
                "Capacitor_Switch", "external_fault","ferroresonance",  "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"]

plt.rcParams.update({'legend.fontsize': 14,
                    'axes.labelsize': 18, 
                    'axes.titlesize': 18,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18})
                    
def test_eval(model, history):

    print("\nTesting ")
    train_curves(history, model._name.replace("_"," "))
    
    # create model analytics using testing data
    pred_probas = model.predict(X_test, verbose = 1)

    y_type = np.argmax(y_test, axis = 1)

    pred_type = np.argmax(pred_probas, axis = 1)

    ###################################################################################################################

    print("\nClassification Report: Fault Type ")
    print(classification_report(y_type, pred_type, target_names = type_names, digits=6))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_type, pred_type))

    print("\nConfusion Matrix: Fault Type ")
    conf_matrix = confusion_matrix(y_type, pred_type)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = type_names, title = model._name.replace("_"," ") + " Fault Type")

    print("\nROC Curve: Fault Type")
    plot_roc(y_test, pred_probas, class_names = type_names, title = model._name.replace("_"," ") +" Fault Type")

    ###################################################################################################################

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


test_eval(DARTS_model, history)