import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # gets rid of harmless error messages
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'  # points to proper cuda library to enable GPU usage

import pickle
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

from keras import backend as K
from keras.layers import (Layer, Input, Reshape, Rescaling, Flatten, Dense, Dropout, TimeDistributed, Conv1D, 
                          Activation, LayerNormalization, Embedding, MultiHeadAttention, Lambda, Add)                       
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves

# number of classes (including a no fault class)
NUM_CLASSES = 46

# training iterations
EPOCHS = 10

try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    # strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU


print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)   # running tensorflow > 2.10 does not support multi-GPU usage on Windows/WSL

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

def build_transformer_model():
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

    model._name = "Transformer_Model"

    return model


with strategy.scope():
    transformer_model = build_transformer_model()

transformer_model.summary()
plot_model(transformer_model, to_file='assets/vanilla_model.png', expand_nested=True, show_shapes=True)


checkpoint_filepath = "assets/cnn_attention_fault_detr_v1_full.h5"    # path to save checkpoint weights

# # uncomment if you want to start the training on pre-existing checkpoint weights
# transformer_model.load_weights(checkpoint_filepath)


# begin training with standard data
X_train, X_test, y_train, y_test = prepare_tensors(norm=False)
transformer_model_history = transformer_model.fit(X_train,
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
transformer_model.load_weights(checkpoint_filepath)
transformer_model_history = transformer_model.fit(X_train,
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


with open('assets/transformer_model_fault_detr_v1_history_full', 'wb') as file_pi:    # path to save model history
    pickle.dump(transformer_model_history.history, file_pi)

with open('assets/transformer_model_fault_detr_v1_history_full', "rb") as file_pi:    # path to load model history
    history = pickle.load(file_pi)

transformer_model.load_weights(checkpoint_filepath)
transformer_model.save('assets/vanilla_transformer_model_v1.keras')  # path to save complete model


test_metrics = transformer_model.evaluate(X_test, y_test)
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


test_eval(transformer_model, history)


# perform explicit predictions
loaded_transformer_model = load_model('assets/vanilla_transformer_model_v1.keras')   # path of complete model

def prediction(input, loaded_model): # shape(726, 3), /file.keras

    single_sample_batch = np.expand_dims(input, axis=0) # makes shape(1, 726, 3) to fit layer shape(None, 726, 3)

    pred = loaded_model.predict_on_batch(single_sample_batch) # predicts for a single data file example (726, 3) --> (1, 726, 3)
    # pred = loaded_model.predict(single_sample_batch) # predict for a whole dataset shape(num_files, 726, 3)
    pred_fault = type_names[np.argmax(pred)]
    print(f'Predicted fault type: {pred_fault}')
