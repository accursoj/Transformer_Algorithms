
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import gc
import shutil
from sklearn.model_selection import train_test_split
import requests

# url = "https://github.com/mendyraul/FPL_Datasets/tree/e175b4fed7db93746d1b7ee9f95b17096237b2d7/PSCAD_datasets/Total_Multiclass/exciting_Class1/"
# data = requests.get(url).content



fault_tags = ["exciting Class1","exciting Class2","exciting Class3","exciting Class4","exciting Class5", "exciting Class6","exciting Class7","exciting Class8","exciting Class9","exciting Class10", "exciting Class11","exciting tt","exciting Classww",
              'series_Class1',
               'series_Class2',
               'series_Class3',
               'series_Class4',
               'series_Class5',
               'series_Class6',
               'series_Class7',
               'series_Class8',
               'series_Class9',
               'series_Class10',
               'series_Class11',
               'series_tt',
               'series_ww',
               'transformer_Class1',
               'transformer_Class2',
               'transformer_Class3',
               'transformer_Class4',
               'transformer_Class5',
               'transformer_Class6',
               'transformer_Class7',
               'transformer_Class8',
               'transformer_Class9',
               'transformer_Class10',
               'transformer_Class11',
               'transformer_tt',
               'transformer_ww',
                "Capacitor_Switch", "external_fault","ferroresonance",  
                "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"] 
                  

def process_fault_tag(fault_tag):
    tags = fault_tag.split("_")
    typ = [0]*46
    typ[int(tags[1])] = 1
    # loc = [0]*15
    # loc[int(tags[2])] = 1
    # print(loc)
    gt = typ #+ loc
    return gt

def process_data():
    X = []
    y = []
    siz = 726
    
    count = 1
    for i in range(0,4):   
        
        if i < 3:
            classes =["exciting_","series_","transformer_","transient_"]
            tags = ["Class1","Class2","Class3","Class4","Class5","Class6","Class7","Class8","Class9","class10","class11","tt","ww"]
            print("\n{}".format(classes[i]))
            for j in tqdm(range(0,13), position=0, leave=True):
                # count = j + (14 * (i-1))
                fault_class_path = "FPL_Datasets/PSCAD_datasets/Total_Multiclass/{}{}/".format(classes[i],tags[j])
                fault_file_names = os.listdir(fault_class_path)
                for k in fault_file_names:
                    fault_file_path = fault_class_path + k
                    fault_signal = pd.read_csv(fault_file_path, header=None)
                    signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
                    X.append(np.stack(signal_is, axis=-1))
                    fault_tag = "1_{}_{}".format(count,(i+1))
                    gt = process_fault_tag(fault_tag)
                    y.append(gt)
                count+=1
        elif i == 3:
            trans_tags = ["capacitor switching", "external fault with CT saturation", "ferroresonance", "magnetising inrush", "non-linear load switching","sympathetic inrush"]
            print("\n{}".format(classes[i]))
            for j in tqdm(range(0,6), position=0, leave=True):
                fault_class_path = "FPL_Datasets/PSCAD_datasets/Total_Multiclass/{}{}/".format(classes[i],trans_tags[j])
                fault_file_names = os.listdir(fault_class_path)
                for k in fault_file_names:
                    fault_file_path = fault_class_path + k
                    fault_signal = pd.read_csv(fault_file_path, header=None)
                    signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
                    X.append(np.stack(signal_is, axis=-1))
                    fault_tag = "1_{}_{}".format(count,(i+1))
                    gt = process_fault_tag(fault_tag)
                    y.append(gt)
                count+=1
            
    
    X =  np.array(X, dtype = np.float32)
    y =  np.array(y)
    print(X)
    print(X.shape)
    print(y)
    print(y.shape)
        
    
    return X, y


X, y = process_data()

print(X)
print(y)

print("No. of exciting Class1: \t", len(y[y[:,1]==1]))
print("No. of exciting Class2: \t", len(y[y[:,2]==1]))
print("No. of exciting Class3: \t", len(y[y[:,3]==1]))
print("No. of exciting Class4: \t", len(y[y[:,4]==1]))
print("No. of exciting Class5: \t", len(y[y[:,5]==1]))
print("No. of exciting Class6: \t", len(y[y[:,6]==1]))
print("No. of exciting Class7: \t", len(y[y[:,7]==1]))
print("No. of exciting Class8: \t", len(y[y[:,8]==1]))
print("No. of exciting Class9: \t", len(y[y[:,9]==1]))
print("No. of exciting Class10: \t", len(y[y[:,10]==1]))
print("No. of exciting Class11: \t", len(y[y[:,11]==1]))
print("No. of exciting tt: \t", len(y[y[:,12]==1]))
print("No. of exciting ww: \t", len(y[y[:,13]==1]))
print("No. of series Class1: \t", len(y[y[:,14]==1]))
print("No. of series Class2: \t", len(y[y[:,15]==1]))
print("No. of series Class3: \t", len(y[y[:,16]==1]))
print("No. of series Class4: \t", len(y[y[:,17]==1]))
print("No. of series Class5: \t", len(y[y[:,18]==1]))
print("No. of series Class6: \t", len(y[y[:,19]==1]))
print("No. of series Class7: \t", len(y[y[:,20]==1]))
print("No. of series Class8: \t", len(y[y[:,21]==1]))
print("No. of series Class9: \t", len(y[y[:,22]==1]))
print("No. of series Class10: \t", len(y[y[:,23]==1]))
print("No. of series Class11: \t", len(y[y[:,24]==1]))
print("No. of series tt: \t", len(y[y[:,25]==1]))
print("No. of series ww: \t", len(y[y[:,26]==1]))
print("No. of transformer Class1: \t", len(y[y[:,27]==1]))
print("No. of transformer Class2: \t", len(y[y[:,28]==1]))
print("No. of transformer Class3: \t", len(y[y[:,29]==1]))
print("No. of transformer Class4: \t", len(y[y[:,30]==1]))
print("No. of transformer Class5: \t", len(y[y[:,31]==1]))
print("No. of transformer Class6: \t", len(y[y[:,32]==1]))
print("No. of transformer Class7: \t", len(y[y[:,33]==1]))
print("No. of transformer Class8: \t", len(y[y[:,34]==1]))
print("No. of transformer Class9: \t", len(y[y[:,35]==1]))
print("No. of transformer Class10: \t", len(y[y[:,36]==1]))
print("No. of transformer Class11: \t", len(y[y[:,37]==1]))
print("No. of transformer tt: \t", len(y[y[:,38]==1]))
print("No. of transformer ww: \t", len(y[y[:,39]==1]))
print("No. of Capacitor_Switch: \t", len(y[y[:,40]==1]))
print("No. of external_fault: \t", len(y[y[:,41]==1]))
print("No. of ferroresonance: \t", len(y[y[:,42]==1]))
print("No. of Magnetic_Inrush: \t", len(y[y[:,43]==1]))
print("No. of Non_Linear_Load_Switch: \t", len(y[y[:,44]==1]))
print("No. of sympathetic_inrush: \t", len(y[y[:,45]==1]))



np.save("FPL_Datasets/ML_TIME/signals_full.npy", X)
np.save("FPL_Datasets/ML_TIME/signals_gts3_full.npy", y)


signals = np.load("FPL_Datasets/ML_TIME/signals_full.npy")
signals_gts = np.load("FPL_Datasets/ML_TIME/signals_gts3_full.npy")
print(signals.shape)
print(signals_gts.shape)

X = []
y = []




for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
    if any(signal_gt[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]):
        noise_count = 10
    elif any(signal_gt[[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]):
        noise_count = 4
    elif any(signal_gt[[12, 26, 39]]):
        noise_count = 2
    elif any(signal_gt[[13, 43, 45]]):
        noise_count = 5
    elif any(signal_gt[[25, 38, 41]]):
        noise_count = 1
    elif signal_gt[40] == 1:
        noise_count = 48
    elif signal_gt[42] == 1:
        noise_count = 12
    elif signal_gt[44] == 1:
        noise_count = 24
    

    for n in range(noise_count):
        X.append(signal)
        y.append(signal_gt)
        
X = np.array(X)
np.random.seed(7)
for i in tqdm(range(X.shape[0])):
    noise = np.random.uniform(-5.0, 5.0, (726, 3)).astype(np.float32)
    X[i] = X[i] + noise
y = np.array(y)


np.save("FPL_Datasets/ML_TIME/vanilla_X_norm.npy", X)
np.save("FPL_Datasets/ML_TIME/vanilla_y_norm.npy", y)
del X, y, signals, signals_gts
gc.collect()