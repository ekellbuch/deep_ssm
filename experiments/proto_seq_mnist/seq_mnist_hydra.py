"""
Writing this seqMNIST experiment file in hydra
"""
import os

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import OmegaConf
import wandb

from deep_ssm.data.basic import load_sequential_mnist_all
from deep_ssm.models.time_series_classifiers import pRNNClassifier

@hydra.main(version_base=None, config_path="../../configs/seq_mnist", config_name="seq_mnist")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    # Set hyperparameters
    hidden_size = cfg.model_cfg.hidden_size
    num_epochs = cfg.trainer_cfg.num_epochs
    print(f"running for {num_epochs} epochs")
    num_iters = cfg.model_cfg.num_iters
    batch_size = cfg.data_cfg.batch_size
    learning_rate = cfg.optimizer_cfg.lr

    use_parallel = cfg.model_cfg.parallel
    offline = not cfg.logger_cfg.use_wandb

    # Initialize wandb logger
    logger = WandbLogger(project="nonlinear_ssm", offline=offline)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Load datasets
    train_loader, val_loader, test_loader = load_sequential_mnist_all(
        batch_size=batch_size, val_split=0.1, seed=cfg.seed)

    # set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS Metal backend
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = pRNNClassifier(
        input_size=1,
        hidden_size=hidden_size,
        num_iters=num_iters,
        parallel=use_parallel,
        learn_diags=cfg.model_cfg.learn_diags,
        k=cfg.model_cfg.k,
        clip=cfg.model_cfg.clip,
        last_only=cfg.model_cfg.last_only,
        learn_initial_states=cfg.model_cfg.learn_initial_states,
        num_classes=10,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    epochs_without_improvement = 0
    best_val_loss = torch.inf
    experiment_dir = f"./checkpoints/{wandb.run.id}"
    os.makedirs(experiment_dir, exist_ok=True)
    best_model_path = os.path.join(experiment_dir, "best_model.pth")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)  # (B, T, D)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if cfg.optimizer_cfg.clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer_cfg.gradient_clip_val)
            optimizer.step()

        # Evaluation Loop
        model.eval()
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
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.trainer_cfg.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
            break

    # Evaluate on Test Set
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    print("Best model loaded for evaluation.")

    model.eval()
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
    print(f"Test Accuracy: {round(100 * test_acc.item(), 2)}%, Test Loss: {test_loss:.4f}")

    # Log Test Metrics to Wandb
    logger.log_metrics({"test_accuracy": test_acc.item(), "test_loss": test_loss})


if __name__ == "__main__":
    main()
