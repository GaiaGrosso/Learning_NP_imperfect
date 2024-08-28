import os
import sys
import datetime
import tensorflow as tf
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
import numpy as np

from NN_v2 import *
from SampleUtils import *
#############################################                                                                                                                            
seed = int(sys.argv[1])
np.random.seed(seed)
print('Random seed:'+str(seed))

DIR_INPUT = sys.argv[2]

N_ref   = 1000000
N_R     = N_ref
N_Bkg   = 200000
N_D     = N_Bkg

N_Data  = DIR_INPUT.split("data")[1]
N_Data  = int(N_Data.split('_')[0])

########## Nuisance parameters ################                                                                                                                         
endcaps_barrel_scale_r = 3
endcaps_barrel_efficiency_r = 1

sigma_sb  = DIR_INPUT.split("_sigmaSB")[-1]
sigma_sb  = float(sigma_sb.split('_')[0])
#sigma_sb  = 0.0005
sigma_se  = endcaps_barrel_scale_r * sigma_sb
sigma_eb  = DIR_INPUT.split("_sigmaEB")[-1]
sigma_eb  = float(sigma_eb.split('_')[0])
sigma_ee  = endcaps_barrel_efficiency_r * sigma_eb
print('sigma_scale: %f, sigma notm: %f'%(sigma_sb, sigma_eb))

scale_barrel       = DIR_INPUT.split("_sb")[-1]
scale_barrel       = float(scale_barrel.split('_')[0])
scale_endcaps      = DIR_INPUT.split("_se")[-1]
scale_endcaps      = float(scale_endcaps.split('_')[0])
efficiency_barrel  = DIR_INPUT.split("_eb")[-1]
efficiency_barrel  = float(efficiency_barrel.split('_')[0])
efficiency_endcaps = DIR_INPUT.split("_ee")[-1]
efficiency_endcaps = float(efficiency_endcaps.split('_')[0])
print(scale_barrel, scale_endcaps, efficiency_barrel, efficiency_endcaps)
N_Data_P = np.random.poisson(lam=N_Data*np.exp(efficiency_barrel), size=1)
N_Data_p = N_Data_P[0]
print('N_Data: '+str(N_Data))
print('N_Data_Pois: '+str(N_Data_p))

###############################################                                                                                                                           
nfile_REF =66
nfile_DATA=22

############ CUTS ##############################                                                                                                                        
M_cut  = 100.
PT_cut = 20.
ETA_cut= 2.4

################################################
####### define output path #####################

OUTPUT_PATH = DIR_INPUT
OUTPUT_FILE_ID = '/Toy'+'_seed'+str(seed)+'_'+str(N_ref)+'ref_'+str(N_Bkg)

# Read data ###################################
#reference+bkg                                                                                                                                               
INPUT_PATH_REF = '/eos/user/g/ggrosso/BSM_Detection/DiLepton_SM/' #'/eos/project/d/dshep/BSM_Detection/DiLepton_SM/'
#INPUT_PATH_DATA = '/eos/user/g/ggrosso/BSM_Detection/DiLepton_EFT06/'#'/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300/'
INPUT_PATH_DATA = '/eos/user/g/ggrosso/PhD/NOTEBOOKS/Paper2/DiLepton_EFT01_new/'
HLF_REF = BuildSample_DY(N_Events=N_ref, INPUT_PATH=INPUT_PATH_REF, seed=seed, nfiles=nfile_REF)
print(HLF_REF.shape)

HLF_BKG = BuildSample_DY(N_Events=N_Data_p, INPUT_PATH=INPUT_PATH_DATA, seed=seed, nfiles=nfile_DATA)
HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_barrel, eta_min=0, eta_max=1.2)
HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_endcaps, eta_min=1.2, eta_max=2.4)
HLF_REF = np.concatenate((HLF_REF, HLF_BKG),axis=0)

target_REF=np.zeros(N_ref)
print('target_REF shape ')
print(target_REF.shape)
target_DATA=np.ones(N_Data_p)
print('target_DATA shape ')
print(target_DATA.shape)
target = np.append(target_REF, target_DATA)
print('target shape ')
print(target.shape)
feature = HLF_REF
print('feature shape ')
print(feature.shape)
#### CUTS #####################################                                                                                                                         
mll = feature[:, -1]
pt1 = feature[:, 0]
pt2 = feature[:, 1]
eta1= feature[:, 2]
eta2= feature[:, 3]
weights = 1*(mll>=M_cut)*(np.abs(eta1)<ETA_cut)*(np.abs(eta2)<ETA_cut)*(pt1>=PT_cut)*(pt2>=PT_cut)                                                                       
feature = feature[weights>0.01]
target  = target[weights>0.01]
weights = weights[weights>0.01]
print('weights shape')
print(weights.shape)

#Apply efficiency modifications #################                                                                                                                              
weights = weights *(target+(N_D*1./N_R)*(1-target))
#weights[target==1] = Apply_Efficiency_Correction_global(weights[target==1],  muon_efficiency=efficiency_barrel)
print(weights.shape)

#remove mass from the inputs ####################                                                                                                                       
target    = np.expand_dims(target, axis=1)
weights   = np.expand_dims(weights, axis=1)
#feature   = feature[:, 0:inputsize]
target    = np.concatenate((target,weights), axis=1 )
#########################################
################ done with data reconstruction

# compute the ideal
HLF_BKG = feature[target[:, 0]==1]
HLF_REF = feature[target[:, 0]==0]
weights_REF = weights[target[:, 0]==0]
MLL  = HLF_BKG[:, -1]
MLL_REF = HLF_REF[:, -1]
bins = np.array([100, 150, 200, 250, 300, 350, 700, 2000])
oi   = plt.hist(MLL, bins=bins)[0]

N_Bkg = plt.hist(MLL_REF, bins=bins, weights=weights_REF[:, 0])[0]#np.sum(weights_REF)
print(oi)
oi = oi[:]#.reshape((-1, 1))
N_Bkg = N_Bkg[:]
print(N_Bkg)
## input coefficients
f = h5py.File('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/MASS_Mcut100_scale_elelike_binnoni.h5', 'r')
a0_nu_s = np.array(f.get('a0'))[:]
a1_nu_s = np.array(f.get('a1'))[:]
a2_nu_s = np.array(f.get('a2'))[:]
f.close()
f = h5py.File('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/MASS_Mcut100_cw_elelike_binnoni_err.h5', 'r')
a0_cw = np.array(f.get('a0'))[:]
a1_cw = np.array(f.get('a1'))[:]*1e-4
a2_cw = np.array(f.get('a2'))[:]*1e-8
f.close()
#a0_cw = np.array([0.993, 0.00495, 0.000401, 0.000085,0.0000264, 0.00000984, 0.00000759, 0.0000014, 0.00000036, 0.00000011 ])
#a1_cw = np.array([7., 5.06, 1.52, 0.67, 0.337, 0.185, 0.23, 0.06, 0.022, 0.0134])*1.e-4
#a2_cw = np.array([1.21, 2.28, 2.2, 1.96, 1.703, 1.432, 2.90, 1.6, 1.09, 0.6769])*1e-5
#a0_cw = a0_cw[1:]
#a1_cw = a1_cw[1:]
#a2_cw = a2_cw[1:]

#def a_i(cw, nu_s, nu_n, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_SM):
#    return (a0_cw+a1_cw*cw+a2_cw*cw**2)*(a0_nu_s+a1_nu_s*nu_s+a2_nu_s*nu_s**2)*np.exp(nu_n)*N_SM/a0_cw/ a0_nu_s
def a_i(cw, nu_s, nu_n, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_SM):
    #return (a0_cw+a1_cw*cw+a2_cw*cw**2)*(a0_nu_s+a1_nu_s*nu_s+a2_nu_s*nu_s**2)*np.exp(nu_n)*N_SM/a0_cw/ a0_nu_s                                                             
    return (a1_cw*cw+a2_cw*cw**2)*N_SM/a0_cw +(a0_nu_s+a1_nu_s*nu_s+a2_nu_s*nu_s**2)*np.exp(nu_n)*200000#/a0_nu_s   
print('0.01, 0.003, 0')
print(a_i(0.01, 0.003, 0, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
print('0.01, 0.0005, 0')
print(a_i(0.01, 0.0005, 0, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
print('0, 0.003, 0')
print(a_i(0., 0.003, 0, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
print('0, 0.0005, 0')
print(a_i(0., 0.0005, 0, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
print('0, 0, 0')
print(a_i(0., 0, 0, a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))

sb0          = np.random.normal(loc=scale_barrel, scale=sigma_sb, size=1)[0]
eb0          = np.random.normal(loc=efficiency_barrel, scale=sigma_eb, size=1)[0]
####### numerator
numerator = MLE_EFT(nu0_s=sb0, nu0_n=eb0, sigma_s=sigma_sb, sigma_n=sigma_eb, N_SM=N_Bkg, luminosity2=200000.,
                    a0_cw=a0_cw, a1_cw=a1_cw, a2_cw=a2_cw, a0_nu_s=a0_nu_s, a1_nu_s=a1_nu_s, a2_nu_s=a2_nu_s,
                     cw_init=0.01, train_cw=True)
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000002)
numerator.compile(loss = MLE_Loss,  optimizer = opt)

print(numerator.summary())                                                                                                                                          
fit_num  = numerator.fit(oi.reshape((1,-1)), oi.reshape((1, -1)), batch_size=oi.shape[0], epochs=30000, verbose=False)

print('MLE: %f, OPT scale: %f, OPT norm: %f, OPT mu: %f'%(-1*fit_num.history['loss'][-1], numerator.nu_s.numpy(), numerator.nu_n.numpy() ,  numerator.cw.numpy() ))
print(a_i(numerator.cw.numpy(), numerator.nu_s.numpy(), numerator.nu_n.numpy(), a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
##### denominator
denominator = MLE_EFT(nu0_s=sb0, nu0_n=eb0, sigma_s=sigma_sb, sigma_n=sigma_eb, N_SM=N_Bkg, luminosity2=200000.,
                      a0_cw=a0_cw, a1_cw=a1_cw, a2_cw=a2_cw, a0_nu_s=a0_nu_s, a1_nu_s=a1_nu_s, a2_nu_s=a2_nu_s,
                      cw_init=0.0, train_cw=False)
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000002)
denominator.compile(loss = MLE_Loss,  optimizer = opt)
print(denominator.summary())
fit_den  = denominator.fit(oi.reshape((1,-1)), oi.reshape((1,-1)), batch_size=oi.shape[0], epochs=30000, verbose=False)

print('MLE: %f, OPT scale: %f, OPT norm: %f'%(-1*fit_den.history['loss'][-1], denominator.nu_s.numpy(), denominator.nu_n.numpy() ))
print(a_i(denominator.cw.numpy(), denominator.nu_s.numpy(), denominator.nu_n.numpy(), a0_cw, a1_cw, a2_cw, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg))
Z = -2*(fit_num.history['loss'][-1]-fit_den.history['loss'][-1])
print(Z, np.sqrt(Z))
'''
# plots
plt.close()
plt.plot(fit_num.history['loss'], label='Numerator')
plt.plot(fit_den.history['loss'], label='Denominator')
plt.legend()
plt.show()
'''
# save output
file_log = OUTPUT_PATH + OUTPUT_FILE_ID+'_ideal_Z.txt'
f = open(file_log, 'w')
f.write(str(Z))
f.close()

file_log = OUTPUT_PATH + OUTPUT_FILE_ID+'_ideal_log.txt'
f = open(file_log, 'w')
f.write('numerator\n')
f.write('loss%f_nu_s%f_nu_n%f_cw%f\n'%(-1*fit_num.history['loss'][-1], numerator.nu_s.numpy(), numerator.nu_n.numpy() ,  numerator.cw.numpy() ))
f.write('denominator\n')
f.write('loss%f_nu_s%f_nu_n%f\n'%(-1*fit_den.history['loss'][-1], denominator.nu_s.numpy(), denominator.nu_n.numpy() ))
f.close()

