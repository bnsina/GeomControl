import matplotlib.pyplot as plt
import argparse
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Jafer\'s rudimentary plotter',
        description='...'
    )
    
    parser.add_argument('algo', nargs='?', help='algorithm')
    parser.add_argument('metric', nargs='?', help='algorithm')
    args = parser.parse_args()
    
    
    file1 = '03-24-2024_18-26_IPp_FINALFULL_PPO_PENALTY_InvertedPendulum-v4_clip_GAE.csv'
    file2 = '03-24-2024_18-53_IPc_FINALFULL_PPO_CLIP_InvertedPendulum-v4_clip_GAE.csv'
    file3 = '03-24-2024_19-14_HCp_FINALFULL_PPO_PENALTY_HalfCheetah-v4_clip_GAE.csv'
    file4 = '03-24-2024_19-43_HCc_FINALFULL_PPO_CLIP_HalfCheetah-v4_clip_GAE.csv'
    
    reward = 'train/reward'
    mc_norm = 'mc_fish_norm'
    mc_trace = 'mc_fish_trace'
    diff = 'mc_emp_diff_norm'
    emp_norm = 'emp_fish_norm'
    emp_trace = 'emp_fish_trace'
    
    select = 1
    
    if select == 0:
        env = 'InvertedPendulum-v4'
    else: 
        env = 'HalfCheetah-v4'
    
    ptitle = f'{env}, {args.algo} w/ GAE: {args.metric}'
    filename = f'{env}_{args.algo}_{args.metric}'
    
    res_df = pd.read_csv(file4, sep=',')
    plt.figure()
    plt.plot(res_df['episode'], res_df[emp_trace])
    plt.xlabel('episode')
    plt.ylabel(args.metric)
    plt.title(ptitle, fontsize=10)
    plt.savefig(filename)
    
    # print(res_df)