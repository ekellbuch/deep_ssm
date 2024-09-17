import lightning as L
import torch
from edit_distance import SequenceMatcher


def calculate_edit_distance_pytorch(seq1, seq2):
  """
     Computes the Levenshtein (edit) distance between two sequences using dynamic programming.
     The edit distance is the minimum number of single-character edits (insertions, deletions, or substitutions)
     required to change one sequence into the other.

     Args:
         seq1 (torch.Tensor): The first sequence, a tensor of shape (n, ) where n is the length of the sequence.
         seq2 (torch.Tensor): The second sequence, a tensor of shape (m, ) where m is the length of the sequence.

     Returns:
         int: The computed Levenshtein distance between seq1 and seq2.

     Notes:
         - This function uses a dynamic programming approach to fill a 2D table (dp) where dp[i, j] represents
           the minimum number of edits required to convert the first i characters of seq1 to the first j characters of seq2.
         - The algorithm has time complexity O(n * m), where n and m are the lengths of seq1 and seq2, respectively.
         - The function assumes that both seq1 and seq2 are 1D integer tensors and processes them element-wise.

     Example:
         seq1 = torch.tensor([1, 2, 3])
         seq2 = torch.tensor([2, 3, 4])
         edit_distance = calculate_edit_distance_pytorch(seq1, seq2)
         print(edit_distance)  # Outputs the edit distance between the two sequences.
  """
  dp = torch.zeros((len(seq1) + 1, len(seq2) + 1), dtype=torch.int32)

  dp[0, :] = torch.arange(len(seq2) + 1)
  dp[:, 0] = torch.arange(len(seq1) + 1)

  for i in range(1, len(seq1) + 1):
    for j in range(1, len(seq2) + 1):
      if seq1[i - 1] == seq2[j - 1]:
        dp[i, j] = dp[i - 1, j - 1]
      else:
        dp[i, j] = 1 + torch.min(dp[i - 1, j], torch.min(dp[i, j - 1], dp[i - 1, j - 1]))

  return dp[len(seq1), len(seq2)].item()


class BCIDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

  def get_lens(self, X_len):
    if self.model.unfolding:
      input_len = ((X_len - self.model.kernelLen) / self.model.strideLen).to(X_len.dtype)
    else:
      input_len = X_len
    return input_len

  def calculate_cer(self, pred, X_len, y, y_len):
    # TODO: optimize this function
    adjustedLens = self.get_lens(X_len)

    total_edit_distance = 0
    total_seq_length = 0
    for iterIdx in range(pred.shape[0]):
      # Decode the predictions
      decodedSeq = torch.argmax(pred[iterIdx, 0: adjustedLens[iterIdx], :], dim=-1)
      decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
      decodedSeq = decodedSeq[decodedSeq != 0]  # Remove blank (0)

      # Get the true sequence
      trueSeq = y[iterIdx][: y_len[iterIdx]]

      # Calculate the edit distance between decodedSeq and trueSeq
      edit_distance = calculate_edit_distance_pytorch(decodedSeq, trueSeq)
      total_edit_distance += edit_distance
      total_seq_length += len(trueSeq)

    # Calculate the Character Error Rate (CER)
    train_cer = total_edit_distance / total_seq_length
    return train_cer


  def training_step(self, batch, batch_idx):
    X, y, X_len, y_len, dayIdx = batch

    input_len = self.get_lens(X_len)

    # forward pass
    pred = self.model(X, dayIdx)

    # calculate CTC loss
    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=input_len,
      target_lengths=y_len,
    )
    self.log("ctc_loss_train", loss)
    return loss

  def _custom_step(self, batch, batch_idx, flag_name='test'):
    X, y, X_len, y_len, dayIdx = batch

    input_len = self.get_lens(X_len)

    # forward pass
    pred = self.model(X, dayIdx)

    # calculate CTC loss
    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=input_len,
      target_lengths=y_len,
    )
    self.log(f"ctc_loss_{flag_name}", loss)

    train_cer = self.calculate_cer(pred, X_len, y, y_len)
    self.log(f"cer_{flag_name}", train_cer)


  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation')

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')



  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")
    if self.args.scheduler_cfg.type == "linear":
      scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        **self.args.scheduler_cfg.configs
      )
    elif self.args.scheduler_cfg.type == "stepLR":
      scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == "cosine_annealing":
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == 'reduce_on_plateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    else:
      raise NotImplementedError(f"Scheduler {self.args.scheduler_cfg.type} not implemented")

    # Copy all scheduler except 'configs' and 'type'
    scheduler_cfg_copy = {k: v for k, v in self.args.scheduler_cfg.items() if k not in ['configs', 'type']}

    scheduler_cfg_copy['scheduler'] = scheduler
    #lr_scheduler_config = { "scheduler" : scheduler}#, "optimizer": optimizer}
    #lr_scheduler_config.update(scheduler_cfg_copy)

    return [optimizer], [scheduler_cfg_copy]
