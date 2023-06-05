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
from wrappers.atari_wrappers import AtariWrapper, PyTorchFrame
from blockworld.blockworld import BlockWorldSingleEnv
from combo_lock.combo_lock import DiabolicalCombinationLock


def get_env(args):
    if args.env.type == "d4rl":
        env = gym.make(args.env.name)
    elif args.env.type == "gridworld":
        env = GridWorld(
            args.env.puzzle_path,
            max_steps=args.env.max_steps,
        )
    elif args.env.type == "atari":
        env = gym.make(args.env.name)
        env = AtariWrapper(env)
        env = PyTorchFrame(env)
    elif args.env.type == "blockworld":
        env = BlockWorldSingleEnv(
            gridlen=args.env.gridlen,
            num_blocks=args.env.num_blocks,
            num_colors=args.env.num_colors,
            blocksize=args.env.blocksize,
        )
    elif args.env.type == "combo_lock":
        env = DiabolicalCombinationLock(args)
    elif args.env.type == "lorl":
        env = LorlWrapper(
            gym.make(args.env.name),
            task=args.env.task,
            use_state=args.env.use_state,
            reward_multiplier=args.env.reward_multiplier,
            sparse=args.env.sparse,
            max_steps=args.env.max_steps,
            normalize=args.env.normalize,
        )
    elif args.env.type == "mujoco":
        env = gym.make(args.env.name)
        if args.env.max_steps:
            env._max_episode_steps = args.env.max_steps
    elif args.env.type == "pushworld":
        pw_env = PushWorldEnv(
            puzzle_path=args.env.puzzle_path, max_steps=args.env.max_steps
        )
        env = PushWorldWrapper(pw_env, use_state=args.env.use_state)
    elif args.env.type == "gym":
        env = gym.make(args.env.name)
        if args.env.max_steps:
            env._max_episode_steps = args.env.max_steps
    else:
        raise NotImplementedError

    if args.env.delay_reward:
        env = DelayedRewardWrapper(env)

    return env


# def flatten(x):
#     """
#     Flattens a list of lists into a numpy array
#     """
#     out = []
#     for episode in x:
#         for item in episode:
#             out.append(item)
#     return np.array(out, dtype=np.float32).squeeze()


def flatten(x):
    """
    Flattens a list of lists into a numpy array
    """
    if not x:
        return np.array([])
    out = []
    for episode in x:
        for item in episode:
            if isinstance(item, np.ndarray) and len(item.shape) == 3:
                # image input
                out.append(item)
            elif isinstance(item, np.ndarray):
                out.append(item.squeeze())
            else:
                out.append(item)
    return np.stack(out, dtype=np.float32)


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


def get_hindsight_logprobs(
    states,
    returns,
    actions,
    h_model,
):
    N = len(states)
    states = torch.from_numpy(np.array(states)).reshape(N, -1).float()
    returns = torch.from_numpy(np.array(returns)).reshape(N, 1).float()
    actions = torch.from_numpy(np.array(actions)).reshape(N, -1).float()
    h_values = h_model.get_hindsight_logprobs(states, returns, actions)
    return h_values.detach().tolist()


def get_density_ratios(states, actions, returns, dd_model):
    states = torch.from_numpy(np.array(states)).float()  # B, D_s
    actions = torch.from_numpy(
        np.array(actions).reshape(-1, dd_model.action_dim)
    ).float()  # B, D_a
    returns = torch.from_numpy(np.array(returns).reshape(-1, 1)).float()  # B, 1
    density_ratios = dd_model.get_density_ratios(states, actions, returns)
    return density_ratios


def get_ret_probs(states, returns, r_model):
    states = torch.from_numpy(np.array(states)).float()  # B, D_s
    returns = torch.from_numpy(np.array(returns).reshape(-1, 1))  # B, 1
    ret_probs = r_model.get_return_probs(states, returns)
    return ret_probs


def assign_hindsight_info(
    buffer, h_model=None, dd_model=None, r_model=None, clip_ratios=True, take_r_product=True
):
    """
    Assigns hindsight logprobs when h_model is passed, otherwise, calculates
    and assigns ratios directly using the dd_model and r_model. If take_r_product is true, the dd_model output must be
    multiplied with the r_model output; if not, the dd_model output can be directly used. This choice is determined by
    how returns are sampled when updating the dd_model.
    """
    if h_model:
        assert (
            not dd_model and not r_model
        ), "Either pass h_model or dd_model and r_model"
        for ep_ind in range(len(buffer)):
            curr_ep_hindsight_logprobs = get_hindsight_logprobs(
                buffer.states[ep_ind],
                buffer.returns[ep_ind],
                buffer.actions[ep_ind],
                h_model,
            )
            buffer.hindsight_logprobs.append(curr_ep_hindsight_logprobs)
    else:
        assert (
            dd_model and r_model
        ), "Either pass h_model or dd_model and r_model"
        for ep_ind in range(len(buffer)):
            curr_ep_density_ratios = get_density_ratios(
                buffer.states[ep_ind],
                buffer.actions[ep_ind],
                buffer.returns[ep_ind],
                dd_model,
            )
            if take_r_product:
                curr_ep_ret_probs = get_ret_probs(
                    buffer.states[ep_ind],
                    buffer.returns[ep_ind],
                    r_model,
                )
                curr_ep_hindsight_ratios = (
                    (curr_ep_density_ratios * curr_ep_ret_probs)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                curr_ep_hindsight_ratios = (
                    curr_ep_density_ratios.detach().cpu().numpy()
                )
            # clipping between 0 and 1. Clipping at 0 is fine but clipping at 1 is a
            # choice to think about because technically the ratios are unbounded above
            # TODO: This should not be required anymore after Sigmoid-ing the DD output?
            if clip_ratios:
                curr_ep_hindsight_ratios = np.clip(
                    curr_ep_hindsight_ratios, a_min=0.0, a_max=1.0
                )
            buffer.hindsight_ratios.append(curr_ep_hindsight_ratios)


def get_hindsight_actions(h_model, states, returns):
    states = np.stack(states).astype(np.float32)
    returns = np.stack(returns).reshape(-1, 1).astype(np.float32)
    states = torch.from_numpy(states)
    returns = torch.from_numpy(returns)
    actions = h_model.get_actions(states, returns)
    return actions


def get_dualdice_update_return_samples(sample_method, r_model, states, r_min, r_max):
    """
    Get return samples to use for the second term in the dualdice loss. If sample_method == "uniform", sample a return
    from Uniform[r_min, r_max]. If sample_method == "r_model", sample the returns from r_model, conditioned on state.
    """
    if sample_method == "uniform":
        num_samples = len(states)
        return_samples = np.random.uniform(low=r_min, high=r_max, size=(num_samples, 1))
    elif sample_method == "r_model":
        states = torch.from_numpy(np.stack(states).astype(np.float32)).to(r_model.device)
        return_dists = r_model.forward(states, return_dists=True)
        return_samples = return_dists.sample().detach().cpu().numpy()
    else:
        raise NotImplementedError

    return return_samples


def get_grad_norm(model):
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach()).to(device) for p in parameters]
            ),
            2.0,
        ).item()
    return total_norm


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def model_init(module, weight_init, bias_init, gain=1):
    """Function taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/utils.py#L53"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
