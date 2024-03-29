import argparse
import datetime
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from hca.model import HCAModel
import pickle


def train(args):
    if args.seed:
        print(
            "============================================================================================"
        )
        print(f"Setting seed: {args.seed}")
        print(
            "============================================================================================"
        )

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load files and create dataloader
    with open(args.data_path, "rb") as f:
        data_dict = pickle.load(f)
    if "act_dim" in data_dict:
        act_dim = data_dict["act_dim"]
    elif "num_acts" in data_dict:
        act_dim = data_dict["num_acts"]
    else:
        raise Exception("Action dim not specified in the data dictionary!")

    continuous = (
        data_dict["continuous"]
        if "continuous" in data_dict
        else args.continuous
    )

    X = torch.from_numpy(data_dict["x"]).float()
    y = torch.from_numpy(data_dict["y"])

    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=True
    )

    dataset_name = (
        args.data_path.split("/")[-2]
        + "_"
        + args.data_path.split("/")[-1].strip(".pkl")
    )
    exp_name = "hca_" + str(args.max_epochs) + "_" + dataset_name

    # Device
    device = torch.device(args.device)

    if args.save_model_freq:
        checkpoint_path = f"checkpoints/{exp_name}_"
        checkpoint_path += f"{datetime.datetime.now().replace(microsecond=0)}"
        setattr(args, "savedir", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

    # Wandb Initialization
    if args.wandb:
        wandb.init(
            name=exp_name,
            project="hca_training",
            config=vars(args),
            entity="ca-exploration",
        )

    # Model
    model = HCAModel(
        X.shape[-1],
        act_dim,
        continuous=continuous,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        activation_fn=args.activation,
        dropout_p=args.dropout,
        lr=args.lr,
        device=device,
    )
    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load(checkpoint)

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    val_results = model.validate(model, val_dataloader)

    if continuous:
        print(
            f"Epoch: 0 | Val Log Likelihood: {round(np.mean(val_results['training/hca_val_logprobs']), 3)}"
        )
    else:
        print(
            f"Epoch: 0 | Val Acc: {round(np.mean(val_results['training/hca_val_acc']), 3)}"
        )

    for epoch in range(args.max_epochs):
        losses = []
        for states, actions in train_dataloader:
            loss, _ = model.train_step(states, actions)
            losses.append(loss)

        if args.wandb:
            wandb.log(
                {
                    "training/avg_loss": np.mean(losses),
                },
                step=epoch,
            )

        print(f"Epoch: {epoch + 1} | Train Loss: {np.mean(losses):.3f}")

        if epoch % args.eval_freq == 0:
            val_results = model.validate(model, val_dataloader)
            if continuous:
                print(
                    f"Epoch: {epoch+1} | Val Log Likelihood: {round(np.mean(val_results['training/hca_val_logprobs']), 3)}"
                )
            else:
                print(
                    f"Epoch: {epoch+1} | Val Acc: {round(np.mean(val_results['training/hca_val_acc']), 3)}"
                )

        # save model weights
        if args.save_model_freq and epoch % args.save_model_freq == 0:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print("saving model at : " + checkpoint_path)
            model.save(f"{checkpoint_path}/model_{epoch}.pt", vars(args))
            print("model saved")
            print(
                "Elapsed Time  : ",
                datetime.datetime.now().replace(microsecond=0) - start_time,
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    ## SAVE MODELS
    if args.save_model_freq:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Final Checkpoint Save!!")
        print("saving model at : " + checkpoint_path)
        model.save(f"{checkpoint_path}/model_{epoch}.pt", vars(args))
        print("model saved")
        print(
            "Elapsed Time  : ",
            datetime.datetime.now().replace(microsecond=0) - start_time,
        )
        print(
            "--------------------------------------------------------------------------------------------"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCA Training Args")

    parser.add_argument(
        "--data-path",
        default="hca_data/ppo_GridWorld-Default:test_v4_2023-01-31 17:12:53/75000_eps.pkl",
        help="path to dataset",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to use wandb logging (default: False)",
    )

    ## Training params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0). 0 means no seeding",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run on (default: cpu)",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=256,
        help="batchsize (default: 256)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="number of hidden layers (default: 2)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="hidden size of models (default:128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for actor network (default: 3e-4)",
    )

    parser.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "tanh"],
        help="hidden layer activations",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="hidden layer dropout probability",
    )

    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=100,
        help="Model save frequency in epochs. Use 0 for no saving (default: 1000000)",
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Override whether the action space is continuous or not.",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5,
        help="How often to run evaluation on model.",
    )

    ## Loading checkpoints:
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="",
        help="path to checkpoint (default: "
        "). Empty string does not load a checkpoint.",
    )

    args = parser.parse_args()
    train(args)
