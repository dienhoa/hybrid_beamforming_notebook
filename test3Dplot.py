from mpl_toolkits.mplot3d import Axes3D
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
mat = io.loadmat('gain.mat')
gain_sameclus = mat['gain_sameclus']
gain_diffclus_crosspol = mat['gain_diffclus_crosspol']
gain_diffclus_samepol = mat['gain_diffclus_samepol']
azi_rot = mat['azi_rot']
ele_rot = mat['ele_rot']
realization = 1000
import seaborn as sns
from matplotlib import cm


angle_number=gain_sameclus.shape[1]
gain_sameclus_mean =  np.zeros((angle_number,int(angle_number/2)),dtype=float)
gain_diffclus_crosspol_mean =  np.zeros((angle_number,int(angle_number/2)),dtype=float)
gain_diffclus_samepol_mean =  np.zeros((angle_number,int(angle_number/2)),dtype=float)
gain_sameclus_var =  np.zeros((angle_number,int(angle_number/2)),dtype=float)
gain_diffclus_crosspol_var =  np.zeros((angle_number,int(angle_number/2)),dtype=float)
gain_diffclus_samepol_var =  np.zeros((angle_number,int(angle_number/2)),dtype=float)

for i in range(angle_number):
    for j in range(int(angle_number/2)):
        gain_sameclus_mean[i,j] = np.mean(gain_sameclus[:,i,j])
        gain_sameclus_var[i,j] = np.var(gain_sameclus[:,i,j])
        
        gain_diffclus_crosspol_mean[i,j] = np.mean(gain_diffclus_crosspol[:,i,j])
        gain_diffclus_crosspol_var[i,j] = np.var(gain_diffclus_crosspol[:,i,j])
        
        gain_diffclus_samepol_mean[i,j] = np.mean(gain_diffclus_samepol[:,i,j])
        gain_diffclus_samepol_var[i,j] = np.var(gain_diffclus_samepol[:,i,j])

total_gain = np.zeros((3,angle_number,int(angle_number/2)))
total_gain[0,:,:]=gain_sameclus_mean
total_gain[1,:,:]=gain_diffclus_crosspol_mean
total_gain[2,:,:]=gain_diffclus_samepol_mean
index_gain_max = np.argmax(total_gain, axis=0)
# print(total_gain[index_gain_max])
gain_max=np.max(total_gain,axis=0)


aazi, eele = np.meshgrid(azi_rot, ele_rot, sparse=False, indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.plot_surface(aazi, eele, gain_max)

plt.show()
