import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from hca_model import HCAModel
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
    with open(args.data_path, 'rb') as f:
        data_dict = pickle.load(f)

    X = torch.from_numpy(data_dict['x'])
    y = torch.from_numpy(data_dict['y'])
    num_actions = data_dict['num_acts']
    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2],)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize)

    exp_name = f"hca:{args.puzzle_path.lstrip('maps/').rstrip('.txt')}"

    # Device
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.save_model_freq:
        checkpoint_path = f"checkpoints/{exp_name}_"
        checkpoint_path += f"{datetime.datetime.now().replace(microsecond=0)}"
        setattr(args, "savedir", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

    # Wandb Initialization
    if args.wandb:
        wandb.init(
            name=exp_name,
            project=args.env_name,
            config=vars(args),
            entity="ca-exploration",
        )

    # Agent
    model = HCAModel(X.shape[-1], num_actions, n_layers=args.n_layers, hidden_size=args.hidden_size)
    model = model.to(device)


    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load(checkpoint)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    for epoch in range(args.max_epochs):
        losses = []
        for states, actions in train_dataloader:
            states = states.to(device)
            actions = actions.to(device)
            preds = model(states)
            loss = loss_fn(preds, actions)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if args.wandb:
            wandb.log(
                {
                    "training/avg_loss": np.mean(losses),
                },
                step=epoch,
            )

        print(f"Epoch: {epoch+1} | Train Loss: {np.mean(losses)}")
        
        if epoch % args.eval_every == 0:
            accuracies = []
            for states, actions in val_dataloader:
                preds = model(states)
                preds = preds.argmax(-1)
                accuracies.append(torch.sum(preds == actions))
            if args.wandb:
                wandb.log(
                    {
                        "val/acc": np.mean(accuracies),
                    },
                    step=epoch,
                )

            print(f"Epoch: {epoch+1} | Val Acc: {np.mean(accuracies)}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCA Training Args")

    parser.add_argument(
        "--data-path",
        default="",
        help="path to dataset (default: )",
    )

    parser.add_argument("--wandb", action="store_true", help="whether to use wandb logging (default: False)")

    ## Training params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0). 0 means no seeding",
    )

    parser.add_argument(
        "--cuda", type=bool, default=False, help="run on CUDA (default: False)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="maxmimum training epochs (default: 100)",
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
        default=64,
        help="hidden size of models (default:64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for actor network (default: 3e-4)",
    )

    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=100,
        help="Model save frequency in epochs. Use 0 for no saving (default: 1000000)",
    )

    parser.add_argument("--eval-freq", type=int, default=5, help="How often to run evaluation on model.")

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
    