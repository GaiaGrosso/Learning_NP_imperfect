import os,h5py
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
    return t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-i','--input',       type=str, help="output  EOS directory", required=True)
    parser.add_argument('-p','--pyscript',     type=str, help="name of python script to execute", required=True)
    args = parser.parse_args()
    DIR_INPUT = args.input
    if not DIR_INPUT.endswith('/'):
        DIR_INPUT = DIR_INPUT+'/'
    label = DIR_INPUT.split('/')[-2]+'ideal/'
    os.system("mkdir %s" %label)
    if not os.path.exists(DIR_INPUT):
        os.system("mkdir %s" %DIR_INPUT)
    title = DIR_INPUT.split('/')[-2]
    if os.path.exists(DIR_INPUT+'out/'):
        seeds = Read_seed_from_h5(DIR_INPUT+'out/', title, extension='_tvalues')
    else:
        seeds = np.arange(100)
    for i in range(len(seeds)):
        joblabel = str(i)
        print('seeds: %i'%(seeds[i]))
        # src file
        script_src = open("%s/%s.src" %(label, joblabel) , 'w')
        script_src.write("#!/bin/bash\n")
        script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh\n")
        script_src.write("python %s/%s %i %s" %(os.getcwd(), args.pyscript, int(seeds[i]), DIR_INPUT))
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
