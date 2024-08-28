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
N_Sig   = DIR_INPUT.split("sig")[-1]
N_Sig   = int(N_Sig.split('_')[0]) 


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

N_Bkg_P = np.random.poisson(lam=N_Bkg*np.exp(efficiency_barrel), size=1)
N_Bkg_p = N_Bkg_P[0]
N_Sig_P = np.random.poisson(lam=N_Sig*np.exp(efficiency_barrel), size=1)
N_Sig_p = N_Sig_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))
print('N_Sig: '+str(N_Sig))
print('N_Sig_Pois: '+str(N_Sig_p))
###############################################                                                                                                                           
nfile_REF=66
nfile_SIG=1

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
INPUT_PATH_SIG = '/eos/user/g/ggrosso/BSM_Detection/DiLepton_Zprime300/'#'/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300/'
HLF_REF = BuildSample_DY(N_Events=N_ref+N_Bkg_p, INPUT_PATH=INPUT_PATH_REF, seed=seed, nfiles=nfile_REF)
print(HLF_REF.shape)

HLF_BKG = HLF_REF[N_ref:, :]
HLF_REF = HLF_REF[:N_ref, :]
HLF_SIG = BuildSample_DY(N_Events=N_Sig_p, INPUT_PATH=INPUT_PATH_SIG, seed=seed, nfiles=nfile_SIG)
HLF_BKG = np.concatenate((HLF_BKG, HLF_SIG), axis=0)

HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_barrel, eta_min=0, eta_max=1.2)
HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_endcaps, eta_min=1.2, eta_max=2.4)
HLF_REF = np.concatenate((HLF_REF, HLF_BKG),axis=0)
print(HLF_REF.shape)

target_REF=np.zeros(N_ref)
print('target_REF shape ')
print(target_REF.shape)
target_DATA=np.ones(N_Bkg_p+N_Sig_p)
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

## input coefficients
f = h5py.File('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/MASS_Mcut100_scale_elelike_SM_Zprime_ideal.h5', 'r')
a0_nu_s = np.array(f.get('a0'))[:-1]
a1_nu_s = np.array(f.get('a1'))[:-1]
a2_nu_s = np.array(f.get('a2'))[:-1]
bins = np.array(f.get('bins'))
f.close()
f = h5py.File('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/MASS_Mcut100_scale_elelike_Zprime300_Zprime_ideal.h5', 'r')
a0_mu = np.array(f.get('a0'))[:-1]
a1_mu = np.array(f.get('a1'))[:-1]
a2_mu = np.array(f.get('a2'))[:-1]
a3_mu = np.array(f.get('a3'))[:-1]
a4_mu = np.array(f.get('a4'))[:-1]
f.close()

# compute the ideal
HLF_BKG = feature[target[:, 0]==1]
HLF_REF = feature[target[:, 0]==0]
weights_REF = weights[target[:, 0]==0]
MLL  = HLF_BKG[:, -1]
MLL_REF = HLF_REF[:, -1]
oi    = plt.hist(MLL, bins=bins)[0]
#N_Bkg = plt.hist(MLL_REF, bins=bins, weights=weights_REF[:, 0])[0]#np.sum(weights_REF)
print(oi)
oi = oi[:-1]#.reshape((-1, 1))
#N_Bkg = N_Bkg[:-1]
#print(N_Bkg)

def a_i(mu, nu_s, nu_n, a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig):
        #return (tf.math.multiply(mu*N_Sig/a0_mu, (a0_mu+a1_mu*nu_s+a2_mu*nu_s**2)) +tf.math.multiply(N_Bkg/a0_nu_s,(a0_nu_s+a1_nu_s*nu_s+a2_nu_s*nu_s**2)))*tf.exp(nu_n)
        return (tf.math.multiply(tf.cast(mu*N_Sig, dtype=tf.float32), tf.cast((a0_mu+a1_mu*nu_s+a2_mu*nu_s**2), dtype=tf.float32)) +tf.math.multiply(tf.cast(N_Bkg, dtype=tf.float32),tf.cast(( a0_nu_s+a1_nu_s*nu_s+a2_nu_s*nu_s**2), dtype=tf.float32)))*tf.exp(nu_n)

print('1, 0.003, 0')
print(a_i(1., 0.003, 0., a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig).numpy())
print('1, 0.0005, 0')
print(a_i(1., 0.0005, 0., a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig).numpy())
print('0, 0.003, 0')
print(a_i(0., 0.003, 0., a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig).numpy())
print('0, 0.0005, 0')
print(a_i(0., 0.0005, 0., a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig).numpy())
print('0, 0, 0')
print(a_i(0., 0., 0., a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig).numpy())

sb0          = np.random.normal(loc=scale_barrel, scale=sigma_sb, size=1)[0]
eb0          = np.random.normal(loc=efficiency_barrel, scale=sigma_eb, size=1)[0]

####### numerator
numerator = MLE_Zprime(nu0_s=sb0, nu0_n=eb0, sigma_s=sigma_sb, sigma_n=sigma_eb, N_SM=N_Bkg, N_Zprime=N_Sig*100, 
                    a0_mu=a0_mu, a1_mu=a1_mu, a2_mu=a2_mu, a3_mu=a3_mu, a4_mu=a4_mu, a0_nu_s=a0_nu_s, a1_nu_s=a1_nu_s, a2_nu_s=a2_nu_s,
                     mu_init=0.01, train_mu=True)
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00000000002)
numerator.compile(loss = MLE_Loss,  optimizer = opt)

print(numerator.summary())                                                                                                                                          
fit_num  = numerator.fit(oi.reshape((1,-1)), oi.reshape((1, -1)), batch_size=oi.shape[0], epochs=300, verbose=False)

print('MLE: %f, OPT scale: %f, OPT norm: %f, OPT mu: %f'%(-1*fit_num.history['loss'][-1], numerator.nu_s.numpy(), numerator.nu_n.numpy() ,  numerator.mu.numpy() ))
print(a_i(numerator.mu.numpy(), numerator.nu_s.numpy(), numerator.nu_n.numpy(), a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig))
##### denominator
denominator = MLE_Zprime(nu0_s=sb0, nu0_n=eb0, sigma_s=sigma_sb, sigma_n=sigma_eb, N_SM=N_Bkg, N_Zprime=N_Sig*100,
                    a0_mu=a0_mu, a1_mu=a1_mu, a2_mu=a2_mu, a3_mu=a3_mu, a4_mu=a4_mu, a0_nu_s=a0_nu_s, a1_nu_s=a1_nu_s, a2_nu_s=a2_nu_s,
                         mu_init=0.0, train_mu=False)
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00000000002)
denominator.compile(loss = MLE_Loss,  optimizer = opt)
print(denominator.summary())
fit_den  = denominator.fit(oi.reshape((1,-1)), oi.reshape((1,-1)), batch_size=oi.shape[0], epochs=300, verbose=False)

print('MLE: %f, OPT scale: %f, OPT norm: %f'%(-1*fit_den.history['loss'][-1], denominator.nu_s.numpy(), denominator.nu_n.numpy() ))
print(a_i(denominator.mu.numpy(), denominator.nu_s.numpy(), denominator.nu_n.numpy(), a0_mu, a1_mu, a2_mu, a0_nu_s, a1_nu_s, a2_nu_s, N_Bkg, N_Sig))
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
f.write('loss%f_nu_s%f_nu_n%f_mu%f\n'%(-1*fit_num.history['loss'][-1], numerator.nu_s.numpy(), numerator.nu_n.numpy() ,  numerator.mu.numpy() ))
f.write('denominator\n')
f.write('loss%f_nu_s%f_nu_n%f\n'%(-1*fit_den.history['loss'][-1], denominator.nu_s.numpy(), denominator.nu_n.numpy() ))
f.close()
