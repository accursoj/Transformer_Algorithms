import numpy as np
from io import StringIO
import os
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import math

file_list = []
#directory = input('Enter Folder Directory: ')
directory = '/Users/josephaccurso/Downloads/Dataset for Transformer & PAR transients/data for transformer and par/transient disturbances/capacitor switching'
for filename in os.listdir(directory):
f = os.path.join(directory, filename)
if os.path.isfile(f):
file_list.append(f)
#print(file_list)

#figure, axis = plt.subplots(3,1)

for i in range(len(file_list)):
#for i in range(15):
data = np.asarray(pd.read_csv(file_list[i], header=None))
time, phaseA, phaseB, phaseC = ([] for i in range(4))

for j in range(len(data)):
time.append(data[j,0])
phaseA.append(data[j,1])
phaseB.append(data[j,2])
phaseC.append(data[j,3])
# plt.xlabel('Time (s)')
# plt.ylabel('Differential Current (A)')
# plt.plot(time,phaseA,label='Phase A')
# plt.plot(time,phaseB,label='Phase B')
# plt.plot(time,phaseC,label='Phase C')

# plt.legend(loc=1)
# plt.title('Capactior Switching Transient Data Example')
# print('Presenting data from: ',file_list[i])
# plt.show()

plt.xlabel('Time (s)')
plt.ylabel('Differential Current (A)')
plt.plot(time,phaseA,label='Phase A')
plt.plot(time,phaseB,label='Phase B')
plt.plot(time,phaseC,label='Phase C')

plt.legend(loc=1)
plt.title('Capactior Switching Transient Data Example')
print('Presenting data from: ',file_list[i])
#plt.show()

# L-2 Norm Equation
def L2(phase):
sum = 0
for i in range(len(data)):
sqd = (data[i,phase])**2
sum += sqd
x2 = np.sqrt(sum)
return x2

# Curve Length Equation
def curve_length(phase):
sum = 0
for i in range(2,len(data)):
dif = (data[i,phase]) - (data[(i-1),phase])
abs = np.abs(dif)
sum += abs
L = sum
return L

# Kurtosis
def kurtosis(phase):
mean_sum = 0
moment_sum4 = 0
moment_sum2 = 0
for i in range(len(data)):
mean_sum += data[i,phase]
mean = mean_sum / (len(data))
for i in range(len(data)):
moment_sum4 += ((data[i,phase] - mean) ** 4)
for i in range(len(data)):
moment_sum2 += ((data[i,phase] - mean) ** 2)
K = (moment_sum4 / len(data)) / ((moment_sum2 / len(data)) ** 2)
return K

# Discrete Fourier Transform
def DFT(phase):
amp = np.fft.fft(data[:,phase])
# amp = np.fft.rfft(data[:,phase])
return np.abs(amp)
# Discrete Fourier Transform Frequencies
def DFT_freq(phase):
freq = np.fft.fftfreq(DFT(phase).size)
return freq

# Discrete Wavelet Transform
def DWT(phase, wavelet):
coeffs = pywt.wavedec(data[:,phase],wavelet,level=2)
return coeffs


L2_arr = np.zeros((len(file_list), 3))
curve_length_arr = np.zeros((len(file_list), 3))
kurtosis_arr = np.zeros((len(file_list), 3))
for i in range(len(file_list)):
data = np.asarray(pd.read_csv(file_list[i], header=None))
for j in range(3):
L2_arr[i,j] = L2(j+1)
curve_length_arr[i,j] = curve_length(j+1)
kurtosis_arr[i,j] = kurtosis(j+1)

#print(L2_arr)
#print(curve_length_arr)
#print(kurtosis_arr)

# plt.xlabel('L2-Energy Norm')
# plt.ylabel('Kurtosis') 
# plt.scatter(L2_arr[:,0],kurtosis_arr[:,0],marker='.',label='Phase A')
# plt.scatter(L2_arr[:,1],kurtosis_arr[:,1],marker='.',label='Phase B')
# plt.scatter(L2_arr[:,2],kurtosis_arr[:,2],marker='.',label='Phase C')
# plt.legend(loc=1)
# plt.title('Kurtosis and L2-Energy Norm of Capacitor Switching Transient Faults')
# plt.show()

# plt.xlabel('L2-Energy Norm')
# plt.ylabel('Curve Length')
# plt.scatter(L2_arr[:,0],curve_length_arr[:,0],marker='.',label='Phase A')
# plt.scatter(L2_arr[:,1],curve_length_arr[:,1],marker='.',label='Phase B')
# plt.scatter(L2_arr[:,2],curve_length_arr[:,2],marker='.',label='Phase C')
# plt.legend(loc=1)
# plt.title('Curve Length and L2-Energy Norm of Capacitor Switching Transient Faults')
# plt.show()

# plt.xlabel('Curve Length')
# plt.ylabel('Kurtosis')
# plt.scatter(curve_length_arr[:,0],kurtosis_arr[:,0],marker='.',label='Phase A')
# plt.scatter(curve_length_arr[:,1],kurtosis_arr[:,1],marker='.',label='Phase B')
# plt.scatter(curve_length_arr[:,2],kurtosis_arr[:,2],marker='.',label='Phase C')
# plt.legend(loc=1)
# plt.title('Kurtosis and Curve Length of Capacitor Switching Transient Faults')
# plt.show()

# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.plot(DFT_freq(1),DFT(1),marker='.',label='Phase A')
# plt.plot(DFT_freq(2),DFT(2),marker='.',label='Phase B')
# plt.plot(DFT_freq(3),DFT(3),marker='.',label='Phase C')
# plt.legend(loc=1)
# plt.title('Discrete Fourier Transform Capcitor Switching Example')
# plt.show()

print(DWT(1,'db1'))
# plt.plot(DWT(1,'db1'))
# plt.show
