"""
Writing this seqMNIST experiment file in hydra
"""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import OmegaConf

from deep_ssm.data.basic import load_sequential_mnist
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
    train_loader = load_sequential_mnist("train", batch_size)
    test_loader = load_sequential_mnist("test", batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluation Loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1)  # (B, T, D)
                predictions = model.predict(x)
                correct += torch.sum(predictions == y)
                total += len(y)

        val_acc = correct / total
        print(f"Epoch {epoch}, Val Accuracy: {round(100 * val_acc.item(), 2)}%")
        logger.log_metrics({"val_accuracy": val_acc.item()}, step=epoch)

    print("Training complete")
    print(f"Using {num_iters} quasi-DEER iterations")


if __name__ == "__main__":
    main()
