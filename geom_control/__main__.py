from . import control
from . import multi_utils
import argparse 
import pandas as pd
import numpy as np
from time import perf_counter
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

# convert param df into namespace list for pool.map(), also append verbose flag
def param_lister(input_df, flags, num_jobs):
    ns_list = []
    for i in range(num_jobs):
        pre_ns = pd.concat([input_df.iloc[i,:], flags]).to_dict()
        ns_list.append(argparse.Namespace(**pre_ns))
    return ns_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GeomControl: Jafer\'s Deep RL Experiment Scheduler/Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        'Performs RL experiments in parallel specified by an input CSV file'      
    )
    
    parser.add_argument('filename', nargs='?', help='filename/path of parameter CSV')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print output to console')
    parser.add_argument('-m', '--multi', dest='multi', action='store_true', help='enable multiprocessing')
    args = parser.parse_args()
    
    flags = pd.Series([args.verbose, args.multi], index=['verbose', 'multi'])
    
    input_df = pd.read_csv(args.filename, sep=',')
    # read_csv uses float64 by default, must cast to float32 for torch optim to work properly
    input_df = input_df.astype({c: np.float32 for c in input_df.select_dtypes(include='float64').columns})
    num_jobs, _ = input_df.shape
    
    t_initial = perf_counter()
    
    # sequential
    if args.multi == False:
        for i in range(num_jobs):
            pre_ns = pd.concat([input_df.iloc[i,:], flags]).to_dict()
            job = control.controller(argparse.Namespace(**pre_ns))
            job.run_control()
     
    # multiprocessing with progress bar for processes (instead of for frames like in the sequential case)      
    if args.multi == True:
        if mp.cpu_count() > 1:
            proc_size = mp.cpu_count() - 1 # max recruited processes
        else:
            proc_size = 1
        
        param_list = param_lister(input_df, flags, num_jobs)
        
        r = process_map(multi_utils.runner, param_list, max_workers=proc_size)
        
    t_final = perf_counter()
    print("Time elapsed: " + str(t_final - t_initial) + ' secs')
    
