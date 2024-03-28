from .utils_ppo_standard import (
    make_env, 
    make_ppo_models, 
    eval_model, 
    get_grad_outer, 
    get_grad_outer_trace, 
    get_param_vector_length,
    avg_log_prob_from_batch,
    get_grad,
    )

from collections import defaultdict
import time

import torch.optim
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger

# TEMPORARY: put all exceptions in a seperate file
class ParamException(Exception):
    pass

class ppo_gae:
    
    def __init__(self, namespace):
        self.args = namespace

    
    def run(self):
        
        algorithm_name = "CLIP_Proximal_Policy_Optimization"
        
        # note to self:
        # 2_000
        # 1_000_000
        # 64
        
        # REQUIRED PARAMS FOR THIS MODULE
        # env_name, frames_per_batch, total_frames, test_interval, num_test_episodes, lrate, 
        # weight_decay, anneal_lr, gamma, mini_batch_size, loss_epochs, gae_lambda, clip_epsilon, anneal_clip_epsilon
        # critic_coef, entropy_coef, loss_critic_type
        
        required_params = {"env_name", "frames_per_batch", "total_frames", "test_interval",
                           "num_test_episodes", "lrate", "weight_decay", "anneal_lr",
                           "gamma", "mini_batch_size", "loss_epochs", "gae_lambda",
                           "clip_epsilon", "anneal_clip_epsilon", "critic_coef", "entropy_coef",
                           "loss_critic_type"}
        
        present_params = vars(self.args).keys()
        
        # check that required params are present in input params
        if not required_params.issubset(vars(self.args).keys()):
            raise ParamException(f"Required parameters for {algorithm_name} not present")
        
        
        #--------------------------------------------------------------------------------#
        env_name = self.args.env_name
        frames_per_batch = self.args.frames_per_batch
        total_frames = self.args.total_frames
        test_interval = self.args.test_interval
        num_test_episodes = self.args.num_test_episodes
        # optimizer params
        lrate = self.args.lrate
        weight_decay = self.args.weight_decay
        anneal_lr = self.args.anneal_lr
        # loss params
        gamma = self.args.gamma
        # mini_batch_size = 64
        mini_batch_size = self.args.mini_batch_size
        # advantage params
        loss_epochs = self.args.loss_epochs
        gae_lambda = self.args.gae_lambda
        clip_epsilon = self.args.clip_epsilon
        anneal_clip_epsilon = self.args.anneal_clip_epsilon
        critic_coef = self.args.critic_coef
        entropy_coef =  self.args.entropy_coef
        loss_critic_type = self.args.loss_critic_type
        #-------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------#
        # jafer params
        sample_size = 20    # sample size for grad calculation
        #-------------------------------------------------------------------------------#
        
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_mini_batches = frames_per_batch // mini_batch_size
        total_network_updates = ((total_frames // frames_per_batch) * loss_epochs * num_mini_batches)
        
        actor, critic = make_ppo_models(env_name)       
        actor, critic = actor.to(device), critic.to(device)
        
        # data collector
        collector = SyncDataCollector(
            create_env_fn=make_env(env_name, device),
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=device,
            max_frames_per_traj=-1,
        )
        
        # (optional) replay buffer:
        sampler = SamplerWithoutReplacement()
        data_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(frames_per_batch),
            sampler=sampler,
            batch_size=mini_batch_size,
        )
        
        # advantage module:
        adv_module = GAE(
            gamma = gamma,
            lmbda=gae_lambda,
            value_network = critic,
            average_gae = False
        )
        
        # loss module:
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            loss_critic_type=loss_critic_type,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=True,
        ) 
        
        # optimizers:
        actor_optim = torch.optim.Adam(actor.parameters(), lr=lrate, eps=1e-5)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=lrate, eps=1e-5)
        
        # logger:
        # the actual torchRL logger isn't working for some reason so this is my manual way
        log_list = defaultdict(list)
        eval_list = defaultdict(list)
        
        
        # create test env.
        test_env = make_env(env_name, device)
        test_env.eval() # sets to eval mode
        
        collected_frames = 0
        num_network_updates = 0
        start_time = time.time()
        
        if self.args.multi == False:
            pbar = tqdm.tqdm(total=total_frames)

        sampling_start = time.time()
        
        losses = TensorDict({}, batch_size=[loss_epochs, num_mini_batches])
        
#----------------- MAIN LOOP------------------------------------------------------------#
        for i, data in enumerate(collector):
            
            sampling_time = time.time() - sampling_start
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            
            if self.args.multi == False:
                pbar.update(data.numel())
            
            # pull log episode lengths and rewards for completed trajectories
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            if len(episode_rewards) > 0:
                episode_length = data["next", "step_count"][data["next", "done"]]
                
                log_list["train/reward"].append(episode_rewards.mean().item()) 
                log_list["train/episode_length"].append(episode_length.sum().item() // len(episode_length))
            else:
                log_list["train/reward"].append(None) 
                log_list["train/episode_length"].append(None)
            
            training_start = time.time()
            
            #-----------EMPIRICAL FISHER NORM/TRACE INIT---------------------
            # trace init
            fisher_index = 0
            avg_empirical_fisher_trace = torch.tensor(0)
            # norm init
            param_size = get_param_vector_length(actor)
            avg_empirical_fisher = torch.empty((param_size, param_size))
            #----------------------------------------------------------------
            
            #---------- MC FISHER INIT---------------------------------------
            mc_fisher_index = 0     # keeping things seperate for comparison
            avg_mc_fisher = torch.empty((param_size, param_size))
            #----------------------------------------------------------------
            
            # optimize for loss_epochs steps
            for j in range(loss_epochs):
                
                # GAE
                with torch.no_grad():
                    data = adv_module(data) # just appends advantage key
                data_reshape = data.reshape(-1)
                 
                # extend buffer
                data_buffer.extend(data_reshape)
                
                for k, batch in enumerate(data_buffer):
                    
                    # pull a batch
                    batch = batch.to(device)

                    # need to understand this --> "Linearly decrease the learning rate and clip epsilon"
                    alpha = 1.0
                    if lrate:
                        alpha = 1 - (num_network_updates / total_network_updates)
                        for group in actor_optim.param_groups:
                            group["lr"] = lrate * alpha
                        for group in critic_optim.param_groups:
                            group["lr"] = lrate * alpha
                    if anneal_clip_epsilon:
                        loss_module.clip_epsilon.copy_(clip_epsilon * alpha)
                    num_network_updates += 1

                    #-------- MC FISHER -------------------------------------
                    if j == (loss_epochs-1):
                        mc_fisher_index += 1
                        nabla_pi_loss = avg_log_prob_from_batch(actor, batch, sample_size)
                        
                        nabla_pi_loss.backward()
                        nabla_pi = get_grad(actor)
                        
                        nabla_outer = torch.outer(nabla_pi, nabla_pi)
                        avg_mc_fisher = (avg_mc_fisher*(mc_fisher_index-1) + nabla_outer) / mc_fisher_index
                        
                        actor_optim.zero_grad()
                    #--------------------------------------------------------
                                              
                    # PPO loss forward
                    loss = loss_module(batch)
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    ).detach()
                    critic_loss = loss["loss_critic"]
                    actor_loss = loss["loss_objective"] + loss["loss_entropy"]
                    
                    # backward
                    actor_loss.backward()
                    critic_loss.backward()
                    
                    #-------- EMPIRICAL FISHER TRACE/NORM -------------------
                    # running average, but only on the last epoch of every collect
                    if j == (loss_epochs-1):
                        grad_outer = get_grad_outer(actor)
                        fisher_index += 1
                        
                        # running average empirical fisher matrix (for norm)
                        avg_empirical_fisher = (avg_empirical_fisher*(fisher_index-1) + grad_outer) / fisher_index
                        
                        # running average trace
                        new_trace = torch.trace(grad_outer)
                        avg_empirical_fisher_trace = (avg_empirical_fisher_trace*(fisher_index-1) + new_trace) / fisher_index
                    #--------------------------------------------------------  
                    
                    # step network weights
                    actor_optim.step()
                    critic_optim.step()
                    actor_optim.zero_grad()
                    critic_optim.zero_grad()
                    
            # log mc fisher norm
            mc_fisher_norm = torch.linalg.norm(avg_mc_fisher).detach().numpy()
            log_list["mc_fish_norm"].append(mc_fisher_norm)
            
            # log mc fisher trace
            mc_fisher_trace = torch.trace(avg_mc_fisher).detach().numpy()
            log_list["mc_fish_trace"].append(mc_fisher_trace)

            # log mc-empirical fisher norm
            mc_emp_diff = torch.linalg.norm(avg_mc_fisher - avg_empirical_fisher).detach().numpy()
            log_list["mc_emp_diff_norm"].append(mc_emp_diff)
            
            # log empirical fisher norm
            empirical_fisher_norm = torch.linalg.norm(avg_empirical_fisher).detach().numpy()
            log_list["emp_fish_norm"].append(empirical_fisher_norm)
            
            # log empirical fisher trace
            log_list["emp_fish_trace"].append(avg_empirical_fisher_trace.detach().numpy())
            
            # log training loss and time
            training_time = time.time() - training_start
            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():
                log_col_name = f"train/{key}"
                log_list[log_col_name].append(value.item())
            
            log_list["train/lr"].append(alpha * lrate)
            log_list["train/sampling_time"].append(sampling_time)
            if anneal_clip_epsilon:
                log_list["train/clip_epsilon"].append(alpha * clip_epsilon)
            else:
                log_list["train/clip_epsilon"].append(clip_epsilon)
            
            # evaluate new model
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                if ((i - 1) * frames_in_batch) // test_interval < (i * frames_in_batch) // test_interval:
                    actor.eval() # set eval mode
                    eval_start = time.time()
                    
                    test_rewards = eval_model(actor, test_env, num_episodes = num_test_episodes)
                    eval_time = time.time() - eval_start
                    eval_list["eval/reward"].append(test_rewards.mean().item())
                    eval_list["eval/time"].append(eval_time)
                    
                    actor.train() # set train mode

            collector.update_policy_weights_()
            sampling_start = time.time()
#---------------------------------------------------------------------------------------#
        
        if self.args.multi == False:
            pbar.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training took {execution_time:.2f} seconds to finish")
        
        return [log_list, eval_list]
                
        
                    
            
        
        
        
         