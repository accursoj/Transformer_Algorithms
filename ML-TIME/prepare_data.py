


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import gc
import shutil
from sklearn.model_selection import train_test_split



# fault_tags = ["No_Fault", "AG", "BG", "CG", "AB", "BC", "AC", "ABG", "BCG", "ACG", "ABC", "ABCG", "HIFA", "HIFB", "HIFC",
#                 "Capacitor_Switch", "Linear_Load_Switch", "Non_Linear_Load_Switch", "Transformer_Switch",
#                 "DG_Switch", "Feeder_Switch", "Insulator_Leakage", "Transformer_Inrush"]  
fault_tags = ["exciting Class1","exciting Class2","exciting Class3","exciting Class4","exciting Class5", "exciting Class6","exciting Class7","exciting Class8","exciting Class9","exciting Class10", "exciting Class11","exciting tt","exciting Classww",
                "Capacitor_Switch", "external_fault","ferroresonance",  
                "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"] 
                  

# # directory = '/Users/raulmendy/Desktop/FPL/Datasets/FPL_Datasets/PSCAD_datasets/Total_Multiclass'
# def load_data(directory):
#     file_list = []
#     for file in os.listdir(directory):
#         f = os.path.join(directory, file)
#         if os.path.isfile(f):
#             file_list.append(f)

#     raw_data = np.asarray(pd.read_csv(f, header=None))
#     data_arr = np.zeros((len(file_list), len(raw_data), 3))
#     for i in range(len(file_list)):
#         raw_data = np.asarray(pd.read_csv(file_list[i], header=None))
#         time_list = []
#         for j in range(len(raw_data)):
#             time_list.append(raw_data[j,0])
#             for k in range(3):
#                 data_arr[i,j,k] = raw_data[j,k+1]
#     return time_list, data_arr

# def process_fault_tag(fault_tag):
#     tags = fault_tag.split("_")
#     typ = [0]*20
#     typ[int(tags[1])] = 1
#     # loc = [0]*15
#     # loc[int(tags[2])] = 1
#     # print(loc)
#     gt = typ #+ loc
#     return gt


# def process_data():
#     X = []
#     y = []
#     siz = 726


                    
#     print("Class 1")
#     for i in tqdm(range(1,37), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class1/"
#         for j in ["ep_back_tfull20_r","ep_back_tfull50_r","ep_back_tfull80_r",
#                   "ep_back_thalf20_r","ep_back_thalf50_r","ep_back_thalf80_r",
#                   "ep_f_tfull20_r","ep_f_tfull80_r",
#                   "ep_f_thalf20_r","ep_f_thalf50_r","ep_f_thalf80_r",
#                   "es_back_tfull20_r","es_back_tfull50_r","es_back_tfull80_r",
#                   "es_back_thalf20_r","es_back_thalf50_r","es_back_thalf80_r",
#                   "es_f_tfull20_r","es_f_tfull50_r","es_f_tfull80_r",
#                   "es_f_thalf20_r","es_f_thalf50_r","es_f_thalf80_r"]:
#                 padded_num = str(i).rjust(5, '0')
#                 fault_file_name = "{}{}_01.txt".format(j,padded_num)  
#                 fault_file_path = fault_bus_path + fault_file_name
#                 fault_signal = pd.read_csv(fault_file_path,header=None)
#                 signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#                 X.append(np.stack(signal_is , axis=-1))
#                 fault_tag = "1_1_{}".format(i)

#                 gt = process_fault_tag(fault_tag)
#                 y.append(gt)
                
#     print("Class 2")
#     for i in tqdm(range(37,73), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class2/"
#         for j in ["ep_back_tfull20_r","ep_back_tfull50_r","ep_back_tfull80_r",
#                   "ep_back_thalf20_r","ep_back_thalf50_r","ep_back_thalf80_r",
#                   "ep_f_tfull20_r","ep_f_tfull80_r",
#                   "ep_f_thalf20_r","ep_f_thalf50_r","ep_f_thalf80_r",
#                   "es_back_tfull20_r","es_back_tfull50_r","es_back_tfull80_r",
#                   "es_back_thalf20_r","es_back_thalf50_r","es_back_thalf80_r",
#                   "es_f_tfull20_r","es_f_tfull50_r","es_f_tfull80_r",
#                   "es_f_thalf20_r","es_f_thalf50_r","es_f_thalf80_r"]:
#                 padded_num = str(i).rjust(5, '0')
#                 fault_file_name = "{}{}_01.txt".format(j,padded_num)  
#                 fault_file_path = fault_bus_path + fault_file_name
#                 fault_signal = pd.read_csv(fault_file_path,header=None)
#                 signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#                 X.append(np.stack(signal_is , axis=-1))
#                 fault_tag = "1_2_{}".format(i)

#                 gt = process_fault_tag(fault_tag)
#                 y.append(gt)
                
#     print("Class 3")
#     for i in tqdm(range(73, 109), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class3/"
#         for j in ["ep_back_tfull20_r","ep_back_tfull50_r","ep_back_tfull80_r",
#                   "ep_back_thalf20_r","ep_back_thalf50_r","ep_back_thalf80_r",
#                   "ep_f_tfull20_r","ep_f_tfull80_r",
#                   "ep_f_thalf20_r","ep_f_thalf50_r","ep_f_thalf80_r",
#                   "es_back_tfull20_r","es_back_tfull50_r","es_back_tfull80_r",
#                   "es_back_thalf20_r","es_back_thalf50_r","es_back_thalf80_r",
#                   "es_f_tfull20_r","es_f_tfull50_r","es_f_tfull80_r",
#                   "es_f_thalf20_r","es_f_thalf50_r","es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)  
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_3_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
            
#     print("Class 4")
#     for i in tqdm(range(109, 145), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class4/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_4_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
            
#     print("Class 5")
#     for i in tqdm(range(145, 181), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class5/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_5_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("Class 6")
#     for i in tqdm(range(181, 217), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class6/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_6_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("Class 7")
#     for i in tqdm(range(217, 253), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class7/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_7_{}".format(i)

#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
            
#     print("Class 8")
#     for i in tqdm(range(253, 289), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class8/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_8_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("Class 9")
#     for i in tqdm(range(289, 325), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class9/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_9_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("Class 10")
#     for i in tqdm(range(325, 361), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_class10/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_10_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("Class 11")
#     for i in tqdm(range(361, 396), position=0, leave=True):
#         fault_bus_path = "Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_class11/"
#         for j in ["ep_back_tfull20_r", "ep_back_tfull50_r", "ep_back_tfull80_r",
#                   "ep_back_thalf20_r", "ep_back_thalf50_r", "ep_back_thalf80_r",
#                   "ep_f_tfull20_r", "ep_f_tfull80_r",
#                   "ep_f_thalf20_r", "ep_f_thalf50_r", "ep_f_thalf80_r",
#                   "es_back_tfull20_r", "es_back_tfull50_r", "es_back_tfull80_r",
#                   "es_back_thalf20_r", "es_back_thalf50_r", "es_back_thalf80_r",
#                   "es_f_tfull20_r", "es_f_tfull50_r", "es_f_tfull80_r",
#                   "es_f_thalf20_r", "es_f_thalf50_r", "es_f_thalf80_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j, padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)]
#             X.append(np.stack(signal_is, axis=-1))
#             fault_tag = "1_11_{}".format(i)
    
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)


#     print("\nCapacitor Switching")
#     for i in tqdm(range(1,60), position=0, leave=True):
#         fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_capacitor switching/"
#         for j in ['cap1f','cap2f','cap3f']:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}_r{}_01.txt".format(j,padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path, header = None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#             X.append(np.stack(signal_is , axis=-1))

#             fault_tag = "1_14_{}".format(i)
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)


#     print("\nTransient External Fault w/ CT Saturation")
#     fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_external fault with CT saturation/"
#     for i in tqdm(range(1,1979), position=0, leave=True):
#         for j in ["ext_b230nov23_r", "ext_b500nov23_r", "ext_f230nov23_r","ext_f230nov23_r"]:
#             padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{}{}_01.txt".format(j,padded_num)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path,header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#             X.append(np.stack(signal_is , axis=-1))
#             fault_tag = "1_15_{}".format(i)
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)

#     print("\nTransient Ferroresonance")
#     for i in tqdm(range(1,240), position=0, leave=True):
#         fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_ferroresonance/"
#         for j in ["ferro_b", "ferro_c"]:
#             # padded_num = str(i).rjust(5, '0')
#             fault_file_name = "{} ({}).txt".format(j,i)
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path,header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#             X.append(np.stack(signal_is , axis=-1))
    
#             fault_tag = "1_16_{}".format(i)
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
            
#     print("\nTransiennt Magnetising Inrush")
#     for i in tqdm(range(1,600), position=0, leave=True):
#         fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_magnetising inrush/"
#         for j in ["inra","inrb"]:
#             fault_file_name = "{} ({}).txt".format(j,i)  
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path,header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#             X.append(np.stack(signal_is , axis=-1))
#             fault_tag = "1_17_{}".format(i)
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
#         for j in ["tx1_bw_inrush-40_r","tx1_bw_inrush-80_r","tx1_bw_inrush0_r","tx1_bw_inrush40_r","tx1_bw_inrush80_r","tx1_fw_inrush-40_r","tx1_fw_inrush-80_r","tx1_fw_inrush0_r","tx1_fw_inrush40_r","tx1_fw_inrush80_r"]:
#             if i < 60:
#                 padded_num = str(i).rjust(5, '0')
#                 fault_file_name = "{}{}_01.txt".format(j,padded_num)  
#                 fault_file_path = fault_bus_path + fault_file_name
#                 fault_signal = pd.read_csv(fault_file_path,header=None)
#                 signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#                 X.append(np.stack(signal_is , axis=-1))
#                 fault_tag = "1_17_{}".format(i)
#                 gt = process_fault_tag(fault_tag)
#                 y.append(gt)
                
    
    
#     print("\nNon Linear Load Switching")
#     for i in tqdm(range(1,336), position=0, leave=True):
#         fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_non-linear load switching/"
#         padded_num = str(i).rjust(5, '0')
#         fault_file_name = "ni_for_r{}_01.txt".format(padded_num)
#         fault_file_path = fault_bus_path + fault_file_name
#         fault_signal = pd.read_csv(fault_file_path, header=None)
#         signal_is = [fault_signal[z].values[0:siz] for z in range(1,4)]
#         X.append(np.stack(signal_is , axis=-1))

#         fault_tag = "1_18_{}".format(i)
#         gt = process_fault_tag(fault_tag)
#         y.append(gt)
        

#     print("\nTransient Sympathetic Inrush")
#     for i in tqdm(range(1,600), position=0, leave=True):
#         fault_bus_path = "/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass/transient_sympathetic inrush/"
#         for j in ["sinra","sinrb"]:
#             fault_file_name = "{} ({}).txt".format(j,i)  
#             fault_file_path = fault_bus_path + fault_file_name
#             fault_signal = pd.read_csv(fault_file_path,header=None)
#             signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#             X.append(np.stack(signal_is , axis=-1))
#             fault_tag = "1_19_{}".format(i)
#             gt = process_fault_tag(fault_tag)
#             y.append(gt)
#         for j in ["tx1_symb-40__r","tx1_symb-80__r","tx1_symb0__r","tx1_symb40__r","tx1_symb80__r","tx1_symf-40__r","tx1_symf-80__r","tx1_symf0__r","tx1_symf40__r","tx1_symf80__r"]:
#             if i < 60:
#                 padded_num = str(i).rjust(5, '0')
#                 fault_file_name = "{}{}_01.txt".format(j,padded_num)  
#                 fault_file_path = fault_bus_path + fault_file_name
#                 fault_signal = pd.read_csv(fault_file_path,header=None)
#                 signal_is = [fault_signal[z].values[:siz] for z in range(1,4)]
#                 X.append(np.stack(signal_is , axis=-1))
#                 fault_tag = "1_19_{}".format(i)
#                 gt = process_fault_tag(fault_tag)
#                 y.append(gt)   
    
    
#     X =  np.array(X, dtype = np.float32)
#     y =  np.array(y)
    

#     return X, y


# X, y = process_data()

# print("No. of exciting Class1: \t", len(y[y[:,1]==1]))
# print("No. of xciting Class2: \t", len(y[y[:,2]==1]))
# print("No. of xciting Class3: \t", len(y[y[:,3]==1]))
# print("No. of exciting Class4: \t", len(y[y[:,4]==1]))
# print("No. of exciting Class5: \t", len(y[y[:,5]==1]))
# print("No. of exciting Class6: \t", len(y[y[:,6]==1]))
# print("No. of xciting Class7: \t", len(y[y[:,7]==1]))
# print("No. of exciting Class8: \t", len(y[y[:,8]==1]))
# print("No. of exciting Class9: \t", len(y[y[:,9]==1]))
# print("No. of exciting Class10: \t", len(y[y[:,10]==1]))
# print("No. of xciting Class11: \t", len(y[y[:,11]==1]))
# print("No. of exciting tt: \t", len(y[y[:,12]==1]))
# print("No. of exciting ww: \t", len(y[y[:,13]==1]))
# print("No. of Capacitor_Switch: \t", len(y[y[:,14]==1]))
# print("No. of xternal_fault: \t", len(y[y[:,15]==1]))
# print("No. of ferroresonance: \t", len(y[y[:,16]==1]))
# print("No. of Magnetic_Inrush: \t", len(y[y[:,17]==1]))
# print("No. of Non_Linear_Load_Switch: \t", len(y[y[:,18]==1]))
# print("No. of ysmpathetic_inrush: \t", len(y[y[:,19]==1]))
# # print()
# # print("No. of No Loc: \t", len(y[y[:,23]==1]))
# # print("No. of Loc 1: \t", len(y[y[:,24]==1]))
# # print("No. of Loc 2: \t", len(y[y[:,25]==1]))
# # print("No. of Loc 3: \t", len(y[y[:,26]==1]))
# # print("No. of Loc 4: \t", len(y[y[:,27]==1]))
# # print("No. of Loc 5: \t", len(y[y[:,28]==1]))
# # print("No. of Loc 6: \t", len(y[y[:,29]==1]))
# # print("No. of Loc 7: \t", len(y[y[:,30]==1]))
# # print("No. of Loc 8: \t", len(y[y[:,31]==1]))
# # print("No. of Loc 9: \t", len(y[y[:,32]==1]))
# # print("No. of Loc 10: \t", len(y[y[:,33]==1]))
# # print("No. of Loc 11: \t", len(y[y[:,34]==1]))
# # print("No. of Loc 12: \t", len(y[y[:,35]==1]))
# # print("No. of Loc 13: \t", len(y[y[:,36]==1]))
# # print("No. of Loc 14: \t", len(y[y[:,37]==1]))


# np.save("signals.npy", X)
# np.save("signals_gts3.npy", y)


signals = np.load("/home/isense/Transformer/FPL_Datasets/ML-TIME/signals.npy")
signals_gts = np.load("/home/isense/Transformer/FPL_Datasets/ML-TIME/signals_gts3.npy")

X = []
y = []

for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
    if any(signal_gt[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19]]): # LG, LL, LLG, LLL, LLLG, HIF, Non_Linear_Load_Switch
        noise_count = 20
    # elif any(signal_gt[[15, 21]]):  # Capacitor_Switch, Insulator_Leakage
    #     noise_count = 10
    # elif signal_gt[16] == 1: # Load_Switch
    #     noise_count = 5
    # elif signal_gt[22] == 1: # Transformer_Inrush
    #     noise_count = 30
    # elif signal_gt[0] == 1: # No Fault
    #     noise_count = 100

    for n in range(noise_count):
        X.append(signal)
        y.append(signal_gt)
        
X = np.array(X)
np.random.seed(7)
for i in tqdm(range(X.shape[0])):
    noise = np.random.uniform(-5.0, 5.0, (726, 3)).astype(np.float32)
    X[i] = X[i] + noise
y = np.array(y)