"""
Writing this seqMNIST experiment file in hydra
"""
import os

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import OmegaConf
import wandb
import pdb

from deep_ssm.data.basic import load_sequential_pixels_all
from deep_ssm.models.time_series_classifiers import pRNNClassifier

@hydra.main(version_base=None, config_path="../../configs/seq_mnist", config_name="seq_mnist")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    def d_from_l(num_layers):
        total_params = 200_000
        D = int(np.sqrt(total_params / (4 * num_layers)))
        return D
    # Set hyperparameters
    if cfg.model_cfg.family=='min_gru':
        hidden_size = d_from_l(cfg.model_cfg.num_layers)
    elif cfg.model_cfg.family=='prnn':
        hidden_size = cfg.model_cfg.hidden_size
    else:
        raise ValueError(f"Unknown family: {cfg.model_cfg.family}")
    num_epochs = cfg.trainer_cfg.num_epochs
    num_iters = cfg.model_cfg.num_iters
    batch_size = cfg.data_cfg.batch_size
    learning_rate = cfg.optimizer_cfg.lr
    num_layers = cfg.model_cfg.num_layers
    use_parallel = cfg.model_cfg.parallel
    offline = not cfg.logger_cfg.use_wandb

    # override if fast_dev_run
    if cfg.trainer_cfg.fast_dev_run:
        hidden_size = 2
        num_epochs = 1
        # offline = True
        num_layers = 2

    while_loop = cfg.model_cfg.while_loop
    if cfg.model_cfg.rt:
        while_loop = True
    # Initialize wandb logger
    logger = WandbLogger(project="nonlinear_ssm", offline=offline)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Load datasets
    train_loader, val_loader, test_loader = load_sequential_pixels_all(
        batch_size=batch_size, val_split=0.1, seed=cfg.seed, explore=cfg.data_cfg.explore, dataset_name=cfg.data_cfg.dataset_name)

    # # set device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")  # macOS Metal backend
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if cfg.data_cfg.dataset_name=='mnist':
        d_input=1
    elif cfg.data_cfg.dataset_name=='cifar':
        d_input=3
    else:
        raise ValueError(f"Unknown dataset: {cfg.data_cfg.dataset_name}")
    # Initialize model
    print(f"hiddensize: {hidden_size}")
    model = pRNNClassifier(
        input_size=d_input,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_iters=num_iters,
        parallel=use_parallel,
        learn_diags=cfg.model_cfg.learn_diags,
        k=cfg.model_cfg.k,
        clip=cfg.model_cfg.clip,
        method=cfg.model_cfg.method,
        last_only=cfg.model_cfg.last_only,
        learn_initial_states=cfg.model_cfg.learn_initial_states,
        while_loop=while_loop,
        quasi=cfg.model_cfg.quasi,
        family=cfg.model_cfg.family,
        use_mean=cfg.model_cfg.use_mean,
        conv=cfg.model_cfg.conv,
        num_classes=10,
    )
    model.to(device)
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.rnn.parameters())
    for p in model.parameters():
        print(p.shape)
    print(f"Number of parameters: {num_params}")
    wandb.config.num_params = num_params
    # wandb.watch(model, log="gradients", log_freq=500)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    epochs_without_improvement = 0
    best_val_loss = torch.inf
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "no_sweep"
    if cfg.model_cfg.family=='prnn':
        best_model_string = f"prnn_iters={cfg.model_cfg.num_iters}_data={cfg.data_cfg.dataset_name}_{wandb.run.id}"
        experiment_dir = f"./checkpoints/{sweep_id}/{best_model_string}"
    elif cfg.model_cfg.family=='min_gru':
        best_model_string = f"mingru_layers={cfg.model_cfg.num_layers}_data={cfg.data_cfg.dataset_name}_{wandb.run.id}"
        experiment_dir = f"./checkpoints/{sweep_id}/{best_model_string}"
    else:
        raise ValueError(f"Unknown family: {cfg.model_cfg.family}")
    os.makedirs(experiment_dir, exist_ok=True)
    best_model_path = os.path.join(experiment_dir, f"{best_model_string}_best_model.pth")
    for epoch in tqdm(range(num_epochs)):
        # train loop
        model.train()
        if cfg.model_cfg.rt:
            model.rnn.parallel = True
        correct, total = 0, 0
        train_loss = 0.0
        # grad_norms = []
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)  # (B, T, D)
            optimizer.zero_grad()
            if cfg.model_cfg.rt:
                num_iters = np.random.randint(3, 785) # hard code the number of iterations for sMNIST
                model.rnn.num_iters = num_iters
            # pdb.set_trace()
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)
            loss = criterion(logits, y)
            if cfg.model_cfg.rt:
                loss = loss * num_iters # weight by number of fixed point iterations used
            correct += torch.sum(predictions == y)
            total += len(y)
            train_loss += loss.item() * x.size(0)
            loss.backward()

            # # Gradient norm monitoring
            # total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            # grad_norms.append(total_grad_norm)

            if cfg.optimizer_cfg.clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer_cfg.grad_clip_val)
            optimizer.step()
        t1 = time.time()
        train_loss /= total
        train_acc = correct / total
        logger.log_metrics(
            {
                "train_accuracy": train_acc.item(),
                "train_loss": train_loss,
                "train_time": t1 - t0,
                # "avg_grad_norm": sum(grad_norms) / len(grad_norms),
            },
            step=epoch,
        )

        # Evaluation Loop
        model.eval()
        if cfg.model_cfg.rt:
            model.rnn.parallel = False
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1)  # (B, T, D)
                logits = model(x)
                predictions = torch.argmax(logits, dim=1)
                loss = criterion(logits, y)
                correct += torch.sum(predictions == y)
                total += len(y)
                val_loss += loss.item() * x.size(0)

        val_loss /= total
        val_acc = correct / total
        print(f"Epoch {epoch}, Val Accuracy: {round(100 * val_acc.item(), 2)}%, Val Loss: {val_loss:.4f}")
        logger.log_metrics({"val_accuracy": val_acc.item(), "val_loss": val_loss}, step=epoch)

        # Early Stopping Logic
        if val_loss < best_val_loss - cfg.trainer_cfg.min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset counter
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.trainer_cfg.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
            break

    # Evaluate on Test Set
    if not cfg.trainer_cfg.fast_dev_run:
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print("Best model loaded for evaluation.")

    model.eval()
    if cfg.model_cfg.rt:
        model.rnn.parallel = False
    test_acc, test_loss = eval_loop(model, test_loader, device, criterion)
    print(f"Test Accuracy: {round(100 * test_acc.item(), 2)}%, Test Loss: {test_loss:.4f}")

    # Evaluate in sequential mode also
    model.rnn.parallel = False
    seq_test_acc, seq_test_loss = eval_loop(model, test_loader, device, criterion)

    # Log Test Metrics to Wandb
    logger.log_metrics({"test_accuracy": test_acc.item(),
                        "test_loss": test_loss,
                        "seq_test_accuracy": seq_test_acc.item(),
                        "seq_test_loss": seq_test_loss})

def eval_loop(model, test_loader, device, criterion):
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)  # (B, T, D)
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)
            loss = criterion(logits, y)
            test_loss += loss.item() * x.size(0)
            correct += torch.sum(predictions == y)
            total += len(y)

    test_loss /= total
    test_acc = correct / total
    return test_acc, test_loss


if __name__ == "__main__":
    main()
