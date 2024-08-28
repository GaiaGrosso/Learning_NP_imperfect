import os
import argparse
import numpy as np
import glob
import os.path
import time

def Read_seed_from_h5(DIR_IN, file_name, extension):
    log_file = DIR_IN+file_name+extension+'.h5'
    #print(log_file)                                                                                                                                             
    tvalues_check = np.array([])
    f = h5py.File(log_file,"r")
    tvalues = np.array(f.get('tvalues'))
    t = np.array([])
    if not 'seeds' in list(f.keys()) and not 'files_id' in  list(f.keys()):
        print('Seeds not found')
        f.close()
        exit()
    if 'seeds' in list(f.keys()):
        t = f.get('seeds')
        t = np.array(t)
    elif 'files_id' in list(f.keys()):
        s = np.array(f.get('files_id'))
        for label in s:
            seed = label.split('seed', 1)[1]
            seed = int(seed.split('_', 1)[0])
            t    = np.append(t, seed)
    f.close()
    print(t.shape)
    return t, tvalues

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output',       type=str, help="output  EOS directory", required=True)
    parser.add_argument('-m','--model',        type=str, help="Parametric model", required=False, default="/")
    parser.add_argument('-p','--pyscript',     type=str, help="name of python script to execute", required=True)
    parser.add_argument('-s','--scale',        type=str, help="scale (unit of sigma)", required=True)
    parser.add_argument('-e','--efficiency',   type=str, help="efficiency (unit of sigma)", required=True)
    parser.add_argument('-d','--dimension',    type=str, help="dimensionality", required=True)
    parser.add_argument('-n','--hiddenlayers', type=str, help="number of hidden layers", required=True)
    parser.add_argument('-l','--latentsize',   type=str, help="layer size", required=True)
    parser.add_argument('-w','--wclip',        type=str, help="weight clipping", required=True)
    parser.add_argument('-r','--runs',         type=str, help="number of epochs", required=True)
    parser.add_argument('-z','--zprime',       type=str, help='number of Zprime events (over 200000 bkg events)', required=False, default="0")
    parser.add_argument('-t', '--toys',        type=str, default = "100", help="number of toys to be processed")
    args = parser.parse_args()
    mydir = args.output
    parametric_model = args.model
    if not mydir.endswith("/"):
        mydir += "/"
    os.system("mkdir %s" %mydir)
    output_folder = parametric_model.split('/')[-2]
    parametric_epochs = parametric_model.split('/')[-1]
    parametric_epochs = parametric_epochs.replace('model_weights', '_')
    parametric_epochs = parametric_epochs.replace('.h5', '')
    output_folder = output_folder +parametric_epochs +'/'
    mydir = mydir +output_folder
    os.system("mkdir %s" %mydir)

    label = args.output.split("/")[-2]+'_'+str(time.time())
    os.system("mkdir %s" %label)

    scale         = args.scale
    efficiency    = args.efficiency
    dimension     = args.dimension 
    hidden_layers = args.hiddenlayers
    latentsize    = args.latentsize
    wclip         = args.wclip
    epochs        = args.runs
    n_sig         = args.zprime
    for i in range(int(args.toys)):
        joblabel = str(i)
        # src file
        script_src = open("%s/%s.src" %(label, joblabel) , 'w')
        script_src.write("#!/bin/bash\n")
        script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh\n")
        script_src.write("python %s/%s %s %s %s %s %s %s %s %s %s %s %s %s" %(os.getcwd(), args.pyscript, mydir, joblabel, parametric_model, scale, efficiency, dimension, latentsize, hidden_layers, wclip, epochs, n_sig, joblabel))
        script_src.close()
        os.system("chmod a+x %s/%s.src" %(label, joblabel))
        
        # condor file
        script_condor = open("%s/%s.condor" %(label, joblabel) , 'w')
        script_condor.write("executable = %s/%s.src\n" %(label, joblabel))
        script_condor.write("universe = vanilla\n")
        script_condor.write("output = %s/%s.out\n" %(label, joblabel))
        script_condor.write("error =  %s/%s.err\n" %(label, joblabel))
        script_condor.write("log = %s/%s.log\n" %(label, joblabel))
        script_condor.write("+MaxRuntime = 500000\n")
        script_condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
        script_condor.write("queue\n")
        
        script_condor.close()
        # condor file submission
        os.system("condor_submit %s/%s.condor" %(label, joblabel))
            
        #time.sleep(60)
        #script_condor.write('requirements = (OpSysAndVer =?= "SLCern6")\n') #requirements = (OpSysAndVer =?= "CentOS7")  
