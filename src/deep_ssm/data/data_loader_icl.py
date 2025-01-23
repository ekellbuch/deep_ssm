import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class InContextLinearRegressionDataset(Dataset):
    def __init__(
        self,
        n_trajectories=None,  # Set to None for infinite data
        n_w_samples=1024,
        seq_length=40,
        d_input=5,
        d_output=1,
        noise_std=0.0,
        batch_size=1024,
        **kwargs
    ):
        """
        Dataset for in-context linear regression.
        Generates data on-the-fly when n_trajectories is None.
        """
        self.n_trajectories = n_trajectories
        assert seq_length % 2 == 0, "Sequence length must be even"
        self.seq_length = seq_length
        self.d_input = d_input
        assert (
            d_output <= d_input
        ), "Output dimension must be less than or equal to input dimension"
        self.d_output = d_output
        self.noise_std = noise_std
        self.n_w_samples = n_w_samples
        self.batch_size = batch_size

        if self.n_w_samples != 0:
            self.w_samples = self._generate_w_samples(self.n_w_samples)
            # Generate fixed trajectories at init if n_trajectories is specified
            self.trajectories = self._generate_trajectories(
                self.n_trajectories, w_samples=self.w_samples
            )
        else:
            # For infinite data, we'll generate trajectories in __getitem__
            self.trajectories = None
            self._current_batch = None

    def _generate_w_samples(self, n_w_samples):
        """
        Generate n_w_samples weight matrices.
        """
        return torch.randn(n_w_samples, self.d_input, self.d_output)

    def _generate_trajectories(self, n, w_samples=None):
        """
        Generate n trajectories.
        Shape: (n, d_input, seq_length - 1), (n, d_input, seq_length - 1)
        """
        if w_samples is None:
            # Generate random weight matrices for each trajecotry
            w = torch.randn(n, self.d_input, self.d_output)
        else:
            # Use pre-generated weight matrices
            w = w_samples  # (n_w_samples, d_input, d_output)

        # Generate random input sequences
        x = torch.randn(n, self.seq_length // 2, self.d_input)

        # Generate output sequences
        if self.n_w_samples != 0:
            # Reshape w to match the number of trajectories
            assert n % w.shape[0] == 0
            w = w.repeat(n // w.shape[0], 1, 1)
            y = torch.bmm(x, w)  # (n, seq_length // 2, d_output)
        else:
            y = torch.bmm(x, w)  # (n, seq_length // 2, d_output)

        # Add noise if specified
        if self.noise_std > 0:
            y = (
                y + torch.randn_like(y) * self.noise_std
            )  # (n, seq_length // 2, d_output)

        # Generate an empty sequence to fill in
        out = torch.zeros(n, self.seq_length, self.d_input)
        out[:, 0::2, : self.d_input] = x
        out[:, 1::2, : self.d_output] = y
        return out[:, :-1, :].transpose(1, 2), out[:, 1:, :].transpose(1, 2)

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        if self.n_w_samples != 0:
            # Return pre-generated trajectories
            x, y = self.trajectories
            return x[idx], y[idx]
        else:
            # For infinite data, generate trajectories in batches
            # We'll use the idx to determine which batch this belongs to
            batch_idx = idx % self.batch_size
            if batch_idx == 0 or self._current_batch is None:
                # Generate new batch of trajectories
                self._current_batch = self._generate_trajectories(self.batch_size)

            # Return from current batch
            x, y = self._current_batch
            x = x.transpose(1,2)
            y = y.transpose(1,2)
            return x[batch_idx], y[batch_idx]

    def get_loss(self, y_pred, y_true):
        """
        Compute the mean squared error loss between y_pred and y_true.
        """
        return torch.mean(
            (y_pred[:, : self.d_output, 0::2] - y_true[:, : self.d_output, 0::2]) ** 2
        )

    def get_eval_loss(self, y_pred, y_true):
        return torch.mean(
            (y_pred[:, : self.d_output, 0::2] - y_true[:, : self.d_output, 0::2]) ** 2,
            dim=(0, 1),
        )  # (seq_length // 2)


class InContextLinearRegressionDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):

        self.train_dataset = InContextLinearRegressionDataset(
            n_trajectories=self.args.n_trajectories,
            n_w_samples=self.args.n_w_samples,
            seq_length=self.args.seq_length,
            d_input=self.args.d_input,
            d_output=self.args.d_output,
            noise_std=self.args.noise_std,
            batch_size=self.args.batch_size,
        )
        self.val_dataset = InContextLinearRegressionDataset(
            n_trajectories=self.args.n_trajectories,
            n_w_samples=self.args.n_w_samples,
            seq_length=self.args.seq_length,
            d_input=self.args.d_input,
            d_output=self.args.d_output,
            noise_std=self.args.noise_std,
            batch_size=self.args.batch_size,
        )
        self.test_dataset = InContextLinearRegressionDataset(
            n_trajectories=self.args.n_trajectories,
            n_w_samples=self.args.n_w_samples,
            seq_length=self.args.seq_length,
            d_input=self.args.d_input,
            d_output=self.args.d_output,
            noise_std=self.args.noise_std,
            batch_size=self.args.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
