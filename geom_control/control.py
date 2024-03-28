from datetime import datetime
from os import path, getcwd
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

from .algorithms import ppo_standard, ppo_penalty

class NotYetImplementedException(Exception):
    pass

class controller:
    
    def __init__(self, namespace):
        self.args = namespace
        self.now = datetime.now()
        self.now_str = self.now.strftime('%m-%d-%Y_%H-%M')
        
    def run_control(self):
        
        file_name = f'{self.now_str}_{self.args.job_name}_{self.args.algorithm}_{self.args.env_name}_{self.args.loss}_{self.args.advantage}.csv'

        if self.args.algorithm == 'PPO_CLIP':
            job_object = ppo_standard.ppo_gae(self.args)
        elif self.args.algorithm == 'PPO_PENALTY':
            job_object = ppo_penalty.ppo_penalty_gae(self.args)
        else:
            raise NotYetImplementedException(f'Algorithm {self.args.algorithm} not yet implemented')
            
        log_list, eval_list = job_object.run()
        
        # convert training log to dataframe and print if verbose
        log_df = pd.DataFrame.from_dict(log_list)
        if self.args.verbose == True:
            print(log_df)
        
        
        # print evaluation rewards
        for k, v in eval_list.items():
            print(f"{k}: {v[0]}")
        
        # output training log CSV
        pd.DataFrame.to_csv(log_df, path.join(getcwd(), file_name))
        
        
            
        
        
        
        