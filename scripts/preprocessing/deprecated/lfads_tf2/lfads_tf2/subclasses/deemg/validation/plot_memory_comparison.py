import numpy as np
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))


file_1 = 'mprofile_20210413111429.dat'
file_2 = 'mprofile_20210413120957.dat'
file_3 = 'mprofile_20210413135015.dat'

mem_1 = np.genfromtxt(file_1, usecols=1, skip_header=True)
mem_2 = np.genfromtxt(file_2, usecols=1, skip_header=True)
mem_3 = np.genfromtxt(file_3, usecols=1, skip_header=True)
t_1 = np.arange(mem_1.size)*0.1
t_2 = np.arange(mem_2.size)*0.1
t_3 = np.arange(mem_3.size)*0.1
import matplotlib.pyplot as plt

plt.plot(t_1, mem_1, marker='+', markersize=5, label='tshift ON')
plt.plot(t_2, mem_2, marker='+', markersize=5, color='k', label='tshift OFF')
plt.plot(t_3, mem_3, marker='+', markersize=5, color='r', label='tshift ON - modified')
plt.ylabel("Memory (Bytes)")
plt.xlabel("Time (s)")
plt.legend()
import pdb; pdb.set_trace()