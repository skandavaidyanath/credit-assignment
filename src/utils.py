import numpy as np
import torch
from gridworld.gridworld_env import GridWorld
import gym

try:
    import d4rl
except:
    print("D4RL is not installed!")
try:
    import lorl_env
except:
    print("Lorl env is not installed!")
try:
    from pushworld.gym_env import PushWorldEnv
except:
    print("PushWorldEnv is not installed!")
from wrappers.lorl import LorlWrapper
from wrappers.pw_wrapper import PushWorldWrapper
from wrappers.delayed_reward_wrapper import DelayedRewardWrapper

def get_env(args):
    if args.env.type == "d4rl":
        env = gym.make(args.env.name)
    elif args.env.type == "gridworld":
        env = GridWorld(
            args.env.puzzle_path,
            sparse=args.env.sparse,
            max_steps=args.env.max_steps,
        )
    elif args.env.type == "lorl":
        env = LorlWrapper(
            gym.make(args.env.name),
            task=args.env.task,
            use_state=args.env.use_state,
            reward_multiplier=args.env.reward_multiplier,
            binary_reward=args.env.binary_reward,
            max_steps=args.env.max_steps,
            normalize=args.env.normalize,
        )
    elif args.env.type == "mujoco":
        env = gym.make(args.env.name)
    elif args.env.type == "pushworld":
        pw_env = PushWorldEnv(
            puzzle_path=args.env.puzzle_path, max_steps=args.env.max_steps
        )
        env = PushWorldWrapper(pw_env, use_state=args.env.use_state)
    elif args.env.type == "gym":
        env = gym.make(args.env.name)
    else:
        raise NotImplementedError

    if args.env.delay_reward:
        env = DelayedRewardWrapper(env)

    return env


def flatten(x):
    """
    Flattens a list of lists into a numpy array
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return np.array(out, dtype=np.float32).squeeze()


def tensor_flatten(x):
    """
    Flattens a list of lists of tensor with gradients to torch tensor with gradients
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return torch.stack(out).squeeze()


def unflatten(x, ref_x):
    """
    Unflatten a flattened tensor x into a list of iterables, where each list is the same length as the corresponding
    list in ref_x (which is a list of iterables).
    """
    ret = []
    ind = 0
    for arr in ref_x:
        ret.append(x[ind : ind + len(arr)])
        ind += len(arr)
    return ret


def normalized_atan(x):
    return torch.atan(x) / (torch.pi / 2)


def sigmoid(arr, temp):
    return 1 / (1 + np.exp(-arr / temp))


def get_hindsight_logprobs(h_model, states, returns, actions):
    inputs = []
    for state, g in zip(states, returns):
        inputs.append(np.concatenate([state, [g]]))
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).reshape(len(inputs), -1).float()  # B x D
    h_values = h_model.get_hindsight_logprobs(
        inputs, torch.from_numpy(np.array(actions))
    )
    return h_values.detach().tolist()


def assign_hindsight_logprobs(buffer, h_model):
    for ep_ind in range(len(buffer)):
        curr_ep_hindsight_logprobs = get_hindsight_logprobs(
            h_model,
            buffer.states[ep_ind],
            buffer.returns[ep_ind],
            buffer.actions[ep_ind],
        )
        buffer.hindsight_logprobs.append(curr_ep_hindsight_logprobs)

def get_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]),
                                2.0).item()
    return total_norm