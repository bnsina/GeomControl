import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from tensordict import TensorDict

# create and transform environment
def make_env(env_name, device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    
    # normalize and clip observation
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    
    return env

def make_ppo_models(env_name):
    env = make_env(env_name, device="cpu")
    
    # shapes
    state_shape = env.observation_spec["observation"].shape[-1]
    action_shape = env.action_spec.shape[-1]
    
    # define policy output distribution: the policy net will train parameters for this distribution
    policy_distribution = TanhNormal
    policy_distribution_params = {
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
        "tanh_loc": False,
    }
    
    #### POLICY ARCHITECHTURE
    policy_mlp = MLP(
        in_features=state_shape,
        activation_class=torch.nn.Tanh,
        out_features=action_shape,  # predict only loc
        num_cells=[64, 64],
    )

    # initialize weights with ortho tensor
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()
            
    # append a trainable normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            env.action_spec.shape[-1], scale_lb=1e-8
        ),
    )
    
    # inputs observation -> policy mlp -> loc,scale -> distribution -> ['action'] in tensordicts 
    actor = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=env.action_spec),
        distribution_class=policy_distribution,
        distribution_kwargs=policy_distribution_params,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    
    #### VALUE ARCHITECHTURE
    value_mlp = MLP(
        in_features=state_shape,
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64,64]
    )
    
    # initialize weights with ortho tensor
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # inputs observation -> value
    critic = ValueOperator(
        value_mlp,
        in_keys=["observation"]
    )
    
    return actor, critic


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()

#------------- extra utils ----------------------------------------------------

# average log probability from a sample, to be used as loss for computing
# sum_n nabla pi(a|s)
def avg_log_prob_from_batch(actor, batch, sample_size):
    sampling_dist = actor.build_dist_from_params(batch)
    sample = sampling_dist.sample((sample_size,))
    dist = actor.get_dist(batch)
    return dist.log_prob(sample).exp().sum(dim=1).mean().log()

def get_grad(actor):
    grad_catted = torch.empty(0)
    policy_params = TensorDict.from_module(actor)
    for key, value in policy_params.items(include_nested=True):
        if not isinstance(value, TensorDict):
            grad_catted = torch.cat((grad_catted, value.grad.reshape(-1)), 0)
    del policy_params
    return grad_catted

def get_grad_outer_trace(actor):
    grad_catted = torch.empty(0)
    policy_params = TensorDict.from_module(actor)
    for key, value in policy_params.items(include_nested=True):
        if not isinstance(value, TensorDict):
            grad_catted = torch.cat((grad_catted, value.grad.reshape(-1)), 0)
    del policy_params
    return torch.trace(torch.outer(grad_catted, grad_catted))

def get_grad_outer(actor):
    grad_catted = torch.empty(0)
    policy_params = TensorDict.from_module(actor)
    for key, value in policy_params.items(include_nested=True):
        if not isinstance(value, TensorDict):
            grad_catted = torch.cat((grad_catted, value.grad.reshape(-1)), 0)
    del policy_params
    return torch.outer(grad_catted, grad_catted)

def get_param_vector_length(actor):
    param_size = 0
    policy_params = TensorDict.from_module(actor)
    for key, value in policy_params.items(include_nested=True):
        if not isinstance(value, TensorDict):
            param_size += value.reshape(-1).size(dim=0)
    del policy_params
    return param_size