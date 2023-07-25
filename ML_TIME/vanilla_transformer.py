


import numpy as np
import pandas as pd 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # gets rid of harmless error messages
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import (Layer, Input, Reshape, Rescaling, Flatten, Dense, Dropout, TimeDistributed, Conv1D, 
                          Activation, LayerNormalization, Embedding, MultiHeadAttention, Lambda, Add)
                          
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score
from keras import backend as K

from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import gc

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves

import pickle

try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    # strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

# signals = np.load("FPL_Datasets/ML_TIME/signals_full.npy", mmap_mode="r")
# signals_gts = np.load("FPL_Datasets/ML_TIME/signals_gts3_full.npy", mmap_mode="r")

# X = []
# y = []



# for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
#     if any(signal_gt[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]):
#         noise_count = 10
#     elif any(signal_gt[[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]):
#         noise_count = 4
#     elif any(signal_gt[[12, 26, 39]]):
#         noise_count = 2
#     elif any(signal_gt[[13, 43, 45]]):
#         noise_count = 5
#     elif any(signal_gt[[25, 38, 41]]):
#         noise_count = 1
#     elif signal_gt[40] == 1:
#         noise_count = 48
#     elif signal_gt[42] == 1:
#         noise_count = 12
#     elif signal_gt[44] == 1:
#         noise_count = 24
    
    

#     for n in range(noise_count):
#         X.append(signal)
#         y.append(signal_gt)
        
# X = np.array(X)
# # np.random.seed(7)
# # for i in tqdm(range(X.shape[0])):
# #     noise = np.random.uniform(-1.0, 1.0, (726, 3)).astype(np.float32)
# #     X[i] = X[i] + noise
# y = np.array(y)

# np.save("FPL_Datasets/ML_TIME/vanilla_X_norm.npy", X)
# np.save("FPL_Datasets/ML_TIME/vanilla_y_norm.npy", y)
# del X, y, signals, signals_gts
# gc.collect()

X = np.load("FPL_Datasets/ML_TIME/vanilla_X_norm.npy", mmap_mode="r")
y = np.load("FPL_Datasets/ML_TIME/vanilla_y_norm.npy", mmap_mode="r")
# X = np.load("FPL_Datasets/ML_TIME/signals_full.npy", mmap_mode="r")
# y = np.load("FPL_Datasets/ML_TIME/signals_gts3_full.npy", mmap_mode="r")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
print(X_tr.shape, y_tr.shape)
print(X_te.shape, y_te.shape)

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

print(f'X_train, y_train shapes: {X_train.shape},{y_train.shape}')
print(f'X_test, y_test shapes: {X_test.shape},{y_test.shape}')

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
    input_sig = Input(shape=(726, 3))
    sig = input_sig/6065.3965
    sig = Reshape((6, 121, 3))(sig)
    sig = TimeDistributed(Flatten())(sig)

    sig = Dense(1024, activation="relu")(sig)
    sig = Dropout(0.2)(sig)
    sig = Dense(64, activation="relu")(sig)
    sig = Dropout(0.2)(sig)

    embeddings = Embedding(input_dim=6, output_dim=64)
    position_embed = embeddings(tf.range(start=0, limit=6, delta=1))
    sig = sig + position_embed

    for e in range(4):
        sig = TransformerEncoder(sig, num_heads=4, head_size=64, dropout=0.2, units_dim=64)

    sig = Flatten()(sig)

    typ = Dense(256, activation="relu")(sig)
    typ = Dropout(0.2)(typ)
    typ = Dense(128, activation="relu")(typ)
    typ = Dense(32, activation="relu")(typ)
    typ = Dropout(0.2)(typ)
    typ_output = Dense(46, activation="softmax", name="type")(typ)

    # loc = Dense(256, activation="relu")(sig)
    # loc = Dropout(0.2)(loc)
    # loc = Dense(128, activation="relu")(loc)
    # loc = Dense(32, activation="relu")(loc)
    # loc = Dropout(0.2)(loc)
    # loc_output = Dense(0, activation="softmax", name="loc")(loc)

    model = Model(inputs=input_sig, outputs=[typ_output]) # loc_output])

    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], 
                  optimizer = Adam(learning_rate=0.001),
                  metrics={"type":[ 
                                    CategoricalAccuracy(name="acc"),
                                    MatthewsCorrelationCoefficient(num_classes=46, name ="mcc"),
                                    F1Score(num_classes=46, name='f1_score')
                                  ] #,
                        #    "loc":[
                        #             CategoricalAccuracy(name="acc"),
                        #             MatthewsCorrelationCoefficient(num_classes=15, name ="mcc"),
                        #             F1Score(num_classes=15, name='f1_score')
                        #          ]
                            })

    model._name = "Transformer_Model"

    return model


with strategy.scope():
    transformer_model = build_transformer_model()

checkpoint_filepath = "FPL_Datasets/ML_TIME/cnn_attention_fault_detr_v5_full.h5"
transformer_model.load_weights(checkpoint_filepath)
# transformer_model_model_history = np.load("FPL_Datasets/ML_TIME/transformer_model_fault_detr_v4_history_full.npy", allow_pickle="TRUE").item()
# transformer_model.load_weights(transformer_model_model_history) # second run

transformer_model_history = transformer_model.fit(X_train,
                                                [y_train[:,:46], y_train[:,46:]],
                                                epochs = 150,
                                                batch_size = 64 * strategy.num_replicas_in_sync,
                                                validation_data = (X_test, [y_test[:,:46], y_test[:,46:]]),
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


with open('FPL_Datasets/ML_TIME/transformer_model_fault_detr_v5_history_full', 'wb') as file_pi:
    pickle.dump(transformer_model_history.history, file_pi)

with open('FPL_Datasets/ML_TIME/transformer_model_fault_detr_v5_history_full', "rb") as file_pi:
    history = pickle.load(file_pi)

# np.save("FPL_Datasets/ML_TIME/transformer_model_fault_detr_v5_history_full.npy", transformer_model_history.history)
# transformer_model_model_history = np.load("FPL_Datasets/ML_TIME/transformer_model_fault_detr_v5_history_full.npy", allow_pickle="TRUE").item()
transformer_model.load_weights(checkpoint_filepath) # second run
transformer_model.save('FPL_Datasets/ML_TIME/vanilla_transformer_model.keras')

# transformer_model.load_weights(checkpoint_filepath)

test_metrics = transformer_model.evaluate(X_test, [y_test[:,:46], y_test[:,46:]]) #, y_test[:,46:]])
test_metrics

type_names = ["exciting_Class1","exciting_Class2","exciting_Class3","exciting_Class4","exciting_Class5", "exciting_Class6","exciting_Class7","exciting_Class8","exciting_Class9","exciting_Class10", "exciting_Class11","exciting_tt","exciting_ww",
              'series_Class1','series_Class2','series_Class3','series_Class4','series_Class5','series_Class6','series_Class7','series_Class8','series_Class9','series_Class10','series_Class11','series_tt','series_ww',
               'transformer_Class1','transformer_Class2','transformer_Class3','transformer_Class4','transformer_Class5','transformer_Class6','transformer_Class7','transformer_Class8','transformer_Class9','transformer_Class10','transformer_Class11','transformer_tt','transformer_ww',
                "Capacitor_Switch", "external_fault","ferroresonance",  "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"]
#loc_names = ["No Loc", "Loc 1", "Loc 2", "Loc 3", "Loc 4", "Loc 5", "Loc 6", "Loc 7", "Loc 8", "Loc 9", "Loc 10", "Loc 11", "Loc 12", "Loc 13", "Loc 14"]


plt.rcParams.update({'legend.fontsize': 14,
                    'axes.labelsize': 18, 
                    'axes.titlesize': 18,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18})
                    
def test_eval(model, history):

    print("\nTesting ")
    train_curves(history, model._name.replace("_"," "))
    
    pred_probas = model.predict(X_test, verbose = 1)

    y_type = np.argmax(y_test, axis = 1)
    # y_loc = np.argmax(y_test[:,20:], axis = 1)

    pred_type = np.argmax(pred_probas, axis = 1)
    # pred_loc = np.argmax(pred_probas[1], axis = 1)

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

    # print("\nClassification Report: Fault Location ")
    # print(classification_report(y_loc, pred_loc, target_names = loc_names, digits=6))
    # print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_loc, pred_loc))

    # print("\nConfusion Matrix: Fault Location ")
    # conf_matrix = confusion_matrix(y_loc, pred_loc)
    # test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = loc_names, title = model._name.replace("_"," ") + " Fault Location")

    # print("\nROC Curve: Fault Location")
    # plot_roc(y_test[:,23:], pred_probas[1], class_names = loc_names, title = model._name.replace("_"," ") +" Fault Location")


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# test_eval(transformer_model, transformer_model_history)
test_eval(transformer_model, history)


loaded_transformer_model = load_model('FPL_Datasets/ML_TIME/vanilla_transformer_model.keras')

def prediction(input, loaded_model): # shape(726, 3), /file.keras

    single_sample_batch = np.expand_dims(input, axis=0) # makes shape(1, 726, 3) to fit layer shape(None, 726, 3)

    pred = loaded_model.predict_on_batch(single_sample_batch) # predicts based off a single example
    # pred = loaded_model.predict(single_sample_batch) # predict for a whole dataset shape(num_files, 726, 3)
    # print(f'Input: {X_te[0]}')
    # print(f'True output: {y_te[0]}')
    # print(f'True output name: {type_names[np.argmax(y_te[0])]}')
    # print(f'Predicted probabilities: {pred}')
    print(f'Predicted fault type: {type_names[np.argmax(pred)]}')
