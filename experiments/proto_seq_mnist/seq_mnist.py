"""
Goal is to build an extremely lightweight script for prototyping models using sequential mnist.

TODO:
* Can I apply the prnn object to sequential mnist
* Can I modify prnn to have a learnable diagonal Jacobian instead of one that comes from nonlinear dynamics?
"""

import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger

from deep_ssm.data.basic import load_sequential_mnist
from deep_ssm.models.time_series_classifiers import pRNNClassifier

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of seq_mnist epochs to train for",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="Number of quasi-DEER iterations to use",
    )
    parser.add_argument("--seq", action="store_true", help="evaluate sequentially")
    parser.add_argument("--proto", action="store_true", help="prototype")
    args = parser.parse_args()

    # Set hyperparameters
    input_size = 1
    hidden_size = args.hidden_size  # 256
    num_epochs = args.num_epochs  # 100
    batch_size = 256  # 256
    learning_rate = 0.001
    num_iters = args.num_iters  # 10
    use_parallel = not args.seq
    use_wandb = not args.proto
    if args.proto:
        hidden_size = 16
        num_epochs = 2
    logger = WandbLogger(
        project="nonlinear_ssm",
        offline=args.proto,
    )
    logger.log_hyperparams(vars(args))

    # Load datasets
    train_loader = load_sequential_mnist("train", batch_size)
    test_loader = load_sequential_mnist("test", batch_size)

    # train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pRNNClassifier(input_size=1,
                           hidden_size=hidden_size,
                           num_iters=args.num_iters,
                           parallel=use_parallel,
                           num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs)):
        for x, y in tqdm(train_loader):
            x = x.to(device)  # (B, D, T)
            y = y.to(device)
            x = x.permute(0, 2, 1)  # (B, T, D)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if args.proto:
                break

        # eval loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in tqdm(test_loader):
                x = x.to(device)
                y = y.to(device)
                x = x.permute(0, 2, 1)  # (B,D,T)
                predictions = model.predict(x)
                correct += torch.sum(predictions == y)
                total += len(y)
            print(
                f"Epoch {epoch}, Val Accuracy: {round(100 * (correct/total).item(),2)}%"
            )
            logger.log_metrics({"val_accuracy": (correct / total).item()}, step=epoch)
            # wandb.log({"val_accuracy": round(100 * (correct / total).item(), 2)})
    print("Training complete")
    print(f"Using {num_iters} quasi-DEER iterations")
