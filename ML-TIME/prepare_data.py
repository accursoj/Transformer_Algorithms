# Import necessary libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm  # Provides a progress bar during loops
import gc  # Garbage collector for memory management


# List of fault tags representing different fault classes
fault_tags = ["exciting_Class1","exciting_Class2","exciting_Class3","exciting_Class4","exciting_Class5", "exciting_Class6","exciting_Class7","exciting_Class8","exciting_Class9","exciting_Class10", "exciting_Class11","exciting_tt","exciting_Classww",
              'series_Class1','series_Class2','series_Class3','series_Class4','series_Class5','series_Class6','series_Class7','series_Class8','series_Class9','series_Class10','series_Class11','series_tt','series_ww',
              'transformer_Class1','transformer_Class2','transformer_Class3','transformer_Class4','transformer_Class5','transformer_Class6','transformer_Class7','transformer_Class8','transformer_Class9','transformer_Class10','transformer_Class11','transformer_tt','transformer_ww',
              "Capacitor_Switch", "external_fault","ferroresonance","Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"] 

# Function to process the fault tag and create a one-hot encoded ground truth
def process_fault_tag(fault_tag):
    """Function that takes the applied fault tag for signal, splits the tag into a
    list, and encodes the ground truth label array to tag what class the fault lays"""
    # Split the fault tag by '_' into list to extract class/type information
    tags = fault_tag.split("_")
    typ = [0] * 46  # Initialize an array of zeros with length 46 (number of fault classes)
    typ[int(tags[1])] = 1  # Set the class corresponding to the fault tag to 1
    gt = typ
    return gt

# Function to process the data and create the input (X) and output (y) arrays
def process_data():
    """Opens the txt files containing the 3 signals that classify a fault.
    While the 3 phases are read, they are stacked onto a numpy array and each
    class of fault is tagged for its respective ground truth label.
    Returned: X, y -> list of numpy array
    numpy files signals_full and signals_gts_full saved to augment."""
    X = []  # List to store input data (signal phases from power transformer)
    y = []  # List to store ground truth labels (one-hot encoded fault classes)
    siz = 726  # Desired size of the signal phases (726 iterations of time)

    count = 1
    # Loop through the different fault classes and their types
    for i in range(0, 4):
        if i < 3:
            # Define class labels and types for different faults
            classes =["exciting_","series_","transformer_","transient_"]
            tags = ["Class1","Class2","Class3","Class4","Class5","Class6","Class7","Class8","Class9","class10","class11","tt","ww"]
            print("\n{}".format(classes[i]))
            # Loop through each type of fault within a class 
            for j in tqdm(range(0,13), position=0, leave=True):
                # count = j + (14 * (i-1))
                fault_class_path = "FPL_Datasets/PSCAD_datasets/Total_Multiclass/{}{}/".format(classes[i],tags[j])
                fault_file_names = os.listdir(fault_class_path)
                for k in fault_file_names:
                    fault_file_path = fault_class_path + k #Combine class and directory
                    fault_signal = pd.read_csv(fault_file_path, header=None) #Reading text file with no header
                    signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)] # Extract 3 signal phases and stack them together
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
                    fault_file_path = fault_class_path + k #Combine class and directory
                    fault_signal = pd.read_csv(fault_file_path, header=None) #Reading text file with no header
                    signal_is = [fault_signal[z].values[:siz] for z in range(1, 4)] # Extract 3 signal phases and stack them together
                    X.append(np.stack(signal_is, axis=-1))
                    fault_tag = "1_{}_{}".format(count,(i+1))
                    gt = process_fault_tag(fault_tag)
                    y.append(gt)
                count+=1
    #Format X and y as numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Save the X and y arrays as numpy files
    np.save("FPL_Datasets/ML_TIME/signals_full.npy", X)
    np.save("FPL_Datasets/ML_TIME/signals_gts3_full.npy", y)

    return X, y

def checkData(X,y,fault_tags):
    """Inputs numpy arrays and global fault tags in order to process and check 
    data. Prints shape and values of X and y. Will also print amount of samples
    in each class of faults. """
    # Display the shape of X and y
    print(X.shape)
    print(X)
    print(y.shape)
    print(y)
    # Count the number of samples for each fault class and type
    count = 1
    for x in fault_tags:
        print("Number of {} faults: \n".format(x), len(y[y[:,count]==1]))
        count+=1

def augmentSignals():
    """Loads the saved signals from processdata() in order to augment noise and 
    normalize the signals dependent on their fault tags. Saves the new X and y
    to be used for the transformers. Performs garbage collection."""
    # Load the saved X and y arrays
    signals = np.load("FPL_Datasets/ML_TIME/signals_full.npy")
    signals_gts = np.load("FPL_Datasets/ML_TIME/signals_gts3_full.npy")

    
    
    # Initialize new empty lists to store augmented data (X) and labels (y)
    X = []
    y = []
    # Loop through each signal and its corresponding ground truth label
    # Augment the data by adding noise based on the fault class
    for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
        # Determine the noise count based on the fault class
        if any(signal_gt[[list(range(1,12))]]):
            noise_count = 10
        elif any(signal_gt[[list(range(14,38))]]):
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
    
        # Add augmented data and labels to X and y
        for n in range(noise_count):
            X.append(signal)
            y.append(signal_gt)
    
    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Add random noise to each signal
    np.random.seed(7)
    for i in tqdm(range(X.shape[0])):
        noise = np.random.uniform(-5.0, 5.0, (726, 3)).astype(np.float32)
        X[i] = X[i] + noise
    
    # Save the augmented data and labels as numpy files
    np.save("FPL_Datasets/ML_TIME/X_norm.npy", X)
    np.save("FPL_Datasets/ML_TIME/y_norm.npy", y)
    
    # Clean up memory by deleting variables and performing garbage collection
    del X, y, signals, signals_gts
    gc.collect()
    

if __name__ == "__main__":
    # Call the process_data function to create X and y arrays
    X, y = process_data()
    # Check X and y 
    checkData(X,y,fault_tags)
    #Add noise to signals and perform garbage collection
    augmentSignals()
    #All data saved locally as files
