import torch
from torchaudio.functional import edit_distance

def calculate_cer(pred, adjustedLens, y, y_len):
  """
  Calculate character error rate
  Args:
    pred:
    adjustedLens:
    y:
    y_len:

  Returns:

  """

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
    seq_distance = edit_distance(decodedSeq, trueSeq)
    total_edit_distance += seq_distance
    total_seq_length += len(trueSeq)

  # Calculate the Character Error Rate (CER)
  train_cer = total_edit_distance / total_seq_length
  return train_cer


# Phoneme-to-Word Mapping: Map the phoneme sequence to word candidates: