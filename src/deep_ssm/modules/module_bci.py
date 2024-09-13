import lightning as L
import torch
from edit_distance import SequenceMatcher
import numpy as np


class BCIDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

  def calculate_cer(self, pred, X_len, y, y_len):
    # TODO: optimize this function
    adjustedLens = ((X_len - self.model.kernelLen) / self.model.strideLen).to(
      torch.int32
    )

    total_edit_distance = 0
    total_seq_length = 0
    for iterIdx in range(pred.shape[0]):
      decodedSeq = torch.argmax(
        torch.tensor(pred[iterIdx, 0: adjustedLens[iterIdx], :]),
        dim=-1,
      )  # [num_seq,]
      decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
      decodedSeq = decodedSeq.cpu().detach().numpy()
      decodedSeq = np.array([i for i in decodedSeq if i != 0])

      trueSeq = np.array(
        y[iterIdx][0: y_len[iterIdx]].cpu().detach()
      )

      matcher = SequenceMatcher(
        a=trueSeq.tolist(), b=decodedSeq.tolist()
      )
      total_edit_distance += matcher.distance()
      total_seq_length += len(trueSeq)
    train_cer = total_edit_distance / total_seq_length
    return train_cer


  def training_step(self, batch, batch_idx):
    X, y, X_len, y_len, dayIdx = batch

    # forward pass
    pred = self.model(X, dayIdx)

    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=((X_len - self.args.model_cfg.configs.kernelLen) / self.args.model_cfg.configs.strideLen).to(torch.int32),
      target_lengths=y_len,
    )
    self.log("ctc_loss_train", loss)
    return loss

  def _custom_step(self, batch, batch_idx, flag_name='test'):
    X, y, X_len, y_len, dayIdx = batch

    # forward pass
    pred = self.model(X, dayIdx)

    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=((X_len - self.args.model_cfg.configs.kernelLen) / self.args.model_cfg.configs.strideLen).to(torch.int32),
      target_lengths=y_len,
    )
    self.log(f"ctc_loss_{flag_name}", loss)

    train_cer = self.calculate_cer(pred, X_len, y, y_len)
    self.log(f"eval_cer_{flag_name}", train_cer)


  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation')

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')



  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)

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

    lr_scheduler_config = {
      "scheduler" : scheduler,
      "optimizer": optimizer,
      "interval" : "step",
      "frequency" : 1,
      "name" : None
    }
    return lr_scheduler_config
