import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pywt


file_list = []
directory = ' '
for filename in os.listdir(directory):
    if filename.endswith('.txt'):    
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            file_list.append(f)

for i in range(len(file_list)):
    data = np.asarray(pd.read_csv(file_list[i], header=None))
    time, phaseA, phaseB, phaseC = ([] for i in range(4))

    for j in range(len(data)):
        time.append(data[j,0])
        phaseA.append(data[j,1])
        phaseB.append(data[j,2])
        phaseC.append(data[j,3])

# Discrete Wavelet Transform
def DWT(phase, wavelet, level_val):
    #coeffs = pywt.dwt(data[:,phase],wavelet)
    coeffs = pywt.wavedec(data[:,phase],wavelet,level=level_val)
    return coeffs

dwt_levels = int(input('Input Level of Discrete Wavelet Transform Decomposition: '))
dwt_phase = 1

coeffs = DWT(dwt_phase,'db1',dwt_levels)

def time_rescale(lst, iterations):
    modified_lst = lst.copy()
    modified_lists = []
    for x in range(iterations):
        modified_lst = modified_lst[::2]
        modified_lists.append(modified_lst)
    return modified_lists

res_time_lst = time_rescale(time, dwt_levels)
res_time_lst.reverse()

subplot_rows = dwt_levels + 1

# Plot DWT Decompositions with Shared Y-Axis
fig, axs = plt.subplots(subplot_rows,1,sharey = True,figsize=(10,10))
axs[0].set_title(f'Level {dwt_levels} Approximate Coefficients')
axs[0].plot(res_time_lst[0],coeffs[0])
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Coefficient Values')
for i in range(dwt_levels):
    index = i + 1
    axs[index].set_title(f'Level {dwt_levels - i} Detail Coefficients')
    axs[index].plot(res_time_lst[i],coeffs[index])
    axs[index].set_xlabel('Time')
    axs[index].set_ylabel('Coefficient Values')
plt.tight_layout()
plt.show()

# Plot DWT Decomposition with Independent Y-Axis
fig, axs = plt.subplots(subplot_rows,1,figsize=(10,10))
axs[0].set_title(f'Level {dwt_levels} Approximate Coefficients')
axs[0].plot(res_time_lst[0],coeffs[0])
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Coefficient Values')
for i in range(dwt_levels):
    index = i + 1
    axs[index].set_title(f'Level {dwt_levels - i} Detail Coefficients')
    axs[index].plot(res_time_lst[i],coeffs[index])
    axs[index].set_xlabel('Time')
    axs[index].set_ylabel('Coefficient Values')
plt.tight_layout()
plt.show()

# Plot DWT Decomposition and Original Signal
fig, axs = plt.subplots((subplot_rows+1),1,figsize=(10,10))
axs[0].set_title(f'Phase {dwt_phase} Signal')
axs[0].plot(time,phaseA)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Differential Current')
axs[1].set_title(f'Level {dwt_levels} Approximate Coefficients')
axs[1].plot(res_time_lst[0],coeffs[0])
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Coefficient Values')
for i in range(dwt_levels):
    index = i + 1
    axs[index+1].set_title(f'Level {dwt_levels - i} Detail Coefficients')
    axs[index+1].plot(res_time_lst[i],coeffs[index])
    axs[index+1].set_xlabel('Time')
    axs[index+1].set_ylabel('Coefficient Values')   
plt.tight_layout()
plt.show()

# Plot 3 Phases of Original Signal and DWT Decomposition
fig, axs = plt.subplots((subplot_rows+1),3,figsize=(12,10))
phase_list = [phaseA, phaseB, phaseC]
color_list = ['#1f77b4','#ff7f0e','#2ca02c']
for phase in range(3):
    coeffs = DWT(phase+1,'db1',dwt_levels)
    axs[0,phase].set_title(f'Phase {phase+1} Signal')
    axs[0,phase].plot(time,phase_list[phase],color=color_list[phase])
    axs[0,phase].set_xlabel('Time')
    axs[0,phase].set_ylabel('Differential Current')
    axs[1,phase].set_title(f'Level {dwt_levels} Approximate Coefficients')
    axs[1,phase].plot(res_time_lst[0],coeffs[0],color=color_list[phase])
    axs[1,phase].set_xlabel('Time')
    axs[1,phase].set_ylabel('Coefficient Values')
    for i in range(dwt_levels):
        index = i + 2
        axs[index,phase].set_title(f'Level {dwt_levels - i} Detail Coefficients')
        axs[index,phase].plot(res_time_lst[i],coeffs[index-1],color=color_list[phase])
        axs[index,phase].set_xlabel('Time')
        axs[index,phase].set_ylabel('Coefficient Values')
    plt.tight_layout()
plt.show()