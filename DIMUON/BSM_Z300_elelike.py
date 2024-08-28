import os
import sys
import datetime
import tensorflow as tf
import h5py
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
import numpy as np

from NNUtils import *
from SampleUtils import *
#############################################                                                                                                                            
seed = int(sys.argv[12])#datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

N_ref   = 1000000
N_R     = N_ref
N_Bkg   = 200000
N_D     = N_Bkg
N_Sig   = int(sys.argv[11])
#### Architecture: ###########################  
inputsize   = int(sys.argv[6])
latentsize  = int(sys.argv[7])
n_layers    = int(sys.argv[8])
layers      = [inputsize]

for _ in range(n_layers):
    layers.append(latentsize)
layers.append(1)
print(layers)
hidden_layers = layers[1:-1]
architecture  = ''
for l in hidden_layers:
    architecture += str(l)+'_'

###############################################                                                                                                                         
patience    = 10000
wc          = float(sys.argv[9])
total_epochs= int(sys.argv[10])

########## Nuisance parameters ################                                                                                                                         
endcaps_barrel_scale_r = 3
endcaps_barrel_efficiency_r = 1

sigma_sb  = 0.003
#sigma_sb  = 0.0005
sigma_se  = endcaps_barrel_scale_r * sigma_sb
sigma_eb  = 0.025
sigma_ee  = endcaps_barrel_efficiency_r * sigma_eb

scale_barrel       = float(sys.argv[4]) *sigma_sb
scale_endcaps      = float(sys.argv[4]) *sigma_se
efficiency_barrel  = float(sys.argv[5]) *sigma_eb
efficiency_endcaps = float(sys.argv[5]) *sigma_ee

N_Bkg_P = np.random.poisson(lam=N_Bkg*np.exp(efficiency_barrel), size=1)
N_Bkg_p = N_Bkg_P[0]
N_Sig_P = np.random.poisson(lam=N_Sig*np.exp(efficiency_barrel), size=1)
N_Sig_p = N_Sig_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))
print('N_Bkg: '+str(N_Sig))
print('N_Bkg_Pois: '+str(N_Sig_p))
###############################################                                                                                                                           
nfile_REF=66
nfile_SIG=1

############ CUTS ##############################                                                                                                                        
M_cut  = 100.
PT_cut = 20.
ETA_cut= 2.4

################################################
####### define output path #####################

OUTPUT_PATH = sys.argv[1]
ID ='/Zprime300_'+str(inputsize)+'D_expNfixed_Mcut'+str(M_cut)+'_PTcut'+str(PT_cut)+ '_ETAcut'+str(ETA_cut)
ID+='_sigmaSB'+str(sigma_sb)+'_sigmaEB'+str(sigma_eb)
ID+='_sb'+str(scale_barrel)+'_se'+str(scale_endcaps)+'_eb'+str(efficiency_barrel)+'_ee'+str(efficiency_endcaps)
ID+='_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)
ID+='_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_layers'+str(n_layers)+'_wclip'+str(wc)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/Toy'+str(inputsize)+'D_seed'+str(seed)+'_patience'+str(patience)+'_'+str(N_ref)+'ref_'+str(N_Bkg)

# do not run the job if the toy label is already in the folder                                                                                                            
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
    exit()

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
feature   = feature[:, 0:inputsize]
target    = np.concatenate((target,weights), axis=1 )

#standardize dataset ############################                                                                                                  
mean_list = []
std_list  = []
for j in range(feature.shape[1]):
    vec  = feature[:, j]
    mean = np.mean(vec)
    std  = np.std(vec)
    mean_list.append(mean)
    std_list.append(std)
    if np.min(vec) < 0:
        vec = vec- mean
        vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                    
        vec = vec *1./ mean
    feature[:, j] = vec

#### training ###################################                                                                                                                        
batch_size = feature.shape[0]
print('Batch size:')
print(batch_size)
weights_ParNet = sys.argv[3]

sb0          = np.random.normal(loc=scale_barrel, scale=sigma_sb, size=1)[0]
eb0          = np.random.normal(loc=efficiency_barrel, scale=sigma_eb, size=1)[0]
NUmatrix     = np.array([[0.,            0.            ]])
NURmatrix    = np.array([[0.,            0.            ]])
NU0matrix    = np.array([[sb0,           eb0           ]])
SIGMAmatrix  = np.array([[sigma_sb,      sigma_eb      ]])
mean_pts     = np.array([ mean_list[0],  mean_list[1]   ])
batch_size   = feature.shape[0]
if inputsize != feature.shape[1]:
    print("Number of features doesn't match the iinputsize")
    exit()
# arguments
input_shape     = (None, inputsize)
points          = []
edgebinlist     = []
means           = []
binned_features = []
ParNet_weights  = None

ParNet_weights = weights_ParNet

model        = NPLMupgrade(input_shape=input_shape, N_Bkg=N_Bkg,
                           edgebinlist=edgebinlist,  means=means, points=points, binned_features=binned_features,
                           NUmatrix=NUmatrix, NURmatrix=NURmatrix, NU0matrix=NU0matrix, SIGMAmatrix=SIGMAmatrix,
                           ParNet_weights=ParNet_weights, correction='PAR',
                           architecture=layers, weight_clipping=wc)

model.compile(loss=NPLLoss_New,  optimizer='adam')
print(model.summary())                                                                                                                                          
hist        = model.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=False)
print('End training ')

# metrics #######################################
loss  = np.array(hist.history['loss'])
scale = np.array(hist.history['scale_barrel'])
norm  = np.array(hist.history['efficiency_barrel'])
laux  = np.array(hist.history['Laux'])
print('sb_opt: %f, eb_opt: %f'%(scale[-1], norm[-1]))
# test statistic ################################                                                                                                                      
final_loss = loss[-1]
t_OBS      = -2*final_loss

# save t ########################################             
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()
print('t: %f'%(t_OBS))
# save history ########################                                                                                                   
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs))
keepEpoch   = epoch % patience == 0
f.create_dataset('loss',              data=loss[keepEpoch],    compression='gzip')
f.create_dataset('Laux',              data=laux[keepEpoch],    compression='gzip')
f.create_dataset('scale_barrel',      data=scale[keepEpoch],   compression='gzip')
f.create_dataset('efficiency_barrel', data=norm[keepEpoch],    compression='gzip')
f.create_dataset('epoch',             data=epoch[keepEpoch],   compression='gzip')
f.close()

# save the model ################################ 
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_weights.h5'
model.save_weights(log_weights)

print('----------------------------\n')
print('Find Delta')
total_epochs_d = 200000
if inputsize==2:
    total_epochs_d = 200000
patience_d     = 100
delta          = NPLMupgrade(input_shape=input_shape, N_Bkg=N_Bkg,
                             edgebinlist=edgebinlist,  means=means, points=points, binned_features=binned_features,
                             NUmatrix=NUmatrix, NURmatrix=NURmatrix, NU0matrix=NU0matrix, SIGMAmatrix=SIGMAmatrix,
                             ParNet_weights=ParNet_weights, correction='PAR',
                             architecture=layers, weight_clipping=wc, train_f=False)
print(delta.summary())
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000000005)
delta.compile(loss=NPLLoss_New,  optimizer=opt)
hist = delta.fit(feature, target, batch_size=batch_size, epochs=total_epochs_d, verbose=False)

# metrics #######################################                                                                                                              
loss  = np.array(hist.history['loss'])
scale = np.array(hist.history['scale_barrel'])
norm  = np.array(hist.history['efficiency_barrel'])
laux  = np.array(hist.history['Laux'])
print('sb_opt: %f, eb_opt: %f'%(scale[-1], norm[-1]))

# test statistic ################################                                                                                               
final_loss = loss[-1]
t_OBS      = -2*final_loss
print('Delta: %f'%(t_OBS))

# save t ########################################                                                                                                               
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_deltaPAR.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()

# save history ##################################                                                                              
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_deltaPAR_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_d))
keepEpoch   = epoch % patience_d == 0
f.create_dataset('loss',              data=loss[keepEpoch],    compression='gzip')
f.create_dataset('Laux',              data=laux[keepEpoch],    compression='gzip')
f.create_dataset('scale_barrel',      data=scale[keepEpoch],   compression='gzip')
f.create_dataset('efficiency_barrel', data=norm[keepEpoch],    compression='gzip')
f.create_dataset('epoch',             data=epoch[keepEpoch],   compression='gzip')
f.close()
