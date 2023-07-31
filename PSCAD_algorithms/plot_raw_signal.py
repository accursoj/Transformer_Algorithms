import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ntpath

def plot_signal(file):
    file_name = os.path.splitext(file)[0]
    file_name = ntpath.basename(file_name)
    dataset = np.asarray(pd.read_csv(file, header=None))
    time, phaseA, phaseB, phaseC = ([] for i in range(4))

    for data in range(len(dataset)):
        time.append(dataset[data,0])
        phaseA.append(dataset[data,1])
        phaseB.append(dataset[data,2])
        phaseC.append(dataset[data,3])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Differential Current (A)')
    plt.plot(time,phaseA,label='Phase A')
    plt.plot(time,phaseB,label='Phase B')
    plt.plot(time,phaseC,label='Phase C')
    plt.legend(loc=1)

    plt.title(f'Signal Data from "{file_name}"')
    plt.savefig(f'{file_name}_plot')

plot_signal('FPL_Datasets/PSCAD_datasets/Total_Multiclass/exciting_Class1/ep_back_tfull20_r00002_01.txt')