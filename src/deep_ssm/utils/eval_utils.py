import pickle
from edit_distance import SequenceMatcher  # For original version
import numpy as np
import torch
from deep_ssm.utils.bci_utils import unique_consecutive, ids_to_phonemes
from collections import Counter



def softmax(x, axis=-1):
    """
    Compute the softmax of each element along an axis of x.

    Parameters:
    - x: Input array (can be a 1D or 2D array, etc.)
    - axis: Axis along which the softmax will be computed.
            Default is the last axis.

    Returns:
    - Softmax output array with the same shape as x.
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Exponentiate after subtracting the max
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)  # Sum of exponentials
    return exp_x / sum_exp_x  # Divide by the sum to get softmax values




def count_classes(sequences):
  """
  Takes in a list of sequences of class labels and outputs the class count.

  Args:
  sequences (list of list): A list where each element is a sequence (list) of class labels.

  Returns:
  dict: A dictionary where keys are class labels and values are the count of each class.
  """
  # Flatten the list of sequences into a single list of class labels
  all_classes = [label for sequence in sequences for label in sequence]

  # Use Counter to count occurrences of each class label
  class_counts = Counter(all_classes)

  return dict(class_counts)

def calculate_cer_numpy(pred, adjusted_lens, y, y_len=None):
  # Calculate the adjusted lengths based on model parameters

  total_edit_distance = 0
  total_seq_length = 0

  for iter_idx in range(pred.shape[0]):
    # Decode predicted sequence
    decoded_seq = np.argmax(pred[iter_idx, :adjusted_lens[iter_idx], :], axis=-1)
    # Apply unique consecutive filter and remove zeros (padding)
    decoded_seq = np.array(
      [decoded_seq[i] for i in range(len(decoded_seq)) if i == 0 or decoded_seq[i] != decoded_seq[i - 1]])
    decoded_seq = decoded_seq[decoded_seq != 0]

    # Get true sequence
    if y_len is None:
      true_seq = y[iter_idx]
    else:
      true_seq = y[iter_idx][:y_len[iter_idx]]

    # Calculate edit distance using SequenceMatcher
    matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded_seq.tolist())
    total_edit_distance += matcher.distance()
    total_seq_length += len(true_seq)

  # Calculate and return the Character Error Rate (CER)
  train_cer = total_edit_distance / total_seq_length
  return train_cer



class ModelOutput(object):
  def __init__(self, filename):
    self.filename = filename
    self.model_name = filename.rsplit('/',1)[-1].split("_")[0]
    self.split = filename.rsplit('/',1)[-1].split("_")[-1].split(".")[0]
    self.logits = []
    self.logitLengths = []
    self.trueSeqs = []
    self.transcriptions = []
    self.days = []

    self.register()

  def register(self):
    with open(self.filename, 'rb') as f:
      data = pickle.load(f)

    self.logits = data["logits"] # [[T, C], [T, C], ... [T, C]]
    self.logitLengths = data["logitLengths"]  # [T, T, ...]
    self.trueSeqs = data["trueSeqs"]  # [[S, S, S ]]
    self.transcriptions = data["transcriptions"]  # [sentence, sentence, ...]
    try:
      self.days = data["day"]  # [D, D, ...]
    except:
      pass

  def get_logits(self, idx=None):
    if idx is not None:
      return self.logits[idx][:self.logitLengths[idx]]

    return [logit[:len_logit] for logit, len_logit in zip(self.logits,self.logitLengths)]

  def get_true_labels(self, idx=None):
    if idx is not None:
      return self.trueSeqs[idx]
    return self.trueSeqs

  def get_true_phonemes(self, idx=None):
    true_labels = self.get_true_labels(idx)
    if not(idx is None):
      true_labels = [true_labels]

    all_phonemes = []
    for sample in true_labels:
      phonemes = ids_to_phonemes(sample)
      all_phonemes.append(phonemes)
    return all_phonemes

  def get_transcriptions(self, idx=None):
    if idx is not None:
      return self.transcriptions[idx]
    return self.transcriptions

  def get_sample(self, idx):
    return self.logits[idx], self.logitLengths[idx], self.trueSeqs[idx], self.transcriptions[idx], self.days[idx]


  def calculate_metric(self, metric):
    if metric == 'CER':
      return self.calculate_cer()

    if metric == 'CTC':
      return self.calculate_ctc()

  def calculate_metric_pday(self, metric):
    if metric == 'CER':
      return self.calculate_cer_pday()

    if metric == 'CTC':
      return self.calculate_ctc_pday()

  def calculate_cer(self):
    # calculate CER
    all_cer = 0
    for i in range(len(self.logits)):
      #  len(self.logits) = 880
      #  self.logits[0].shape (56, 41) 56 is T, 41 is Nclasses
      cer = calculate_cer_numpy(self.logits[i][None,...], [self.logitLengths[i]], self.trueSeqs[i][None,...])
      all_cer += cer
    return all_cer / len(self.logits)

  def calculate_cer_pday(self):
    # calculate CER per day
    all_cer = np.zeros(len(np.unique(self.days)))
    counts_pday = np.zeros(len(np.unique(self.days)))
    for i in range(len(self.logits)):
      day_idx = self.days[i]
      cer = calculate_cer_numpy(self.logits[i][None,...], [self.logitLengths[i]], self.trueSeqs[i][None,...])
      all_cer[day_idx] += cer
      counts_pday[day_idx] += 1
    return all_cer / counts_pday

  def calculate_ctc(self):
    all_ctc = 0
    num_seqs = len(self.logits)

    for i in range(num_seqs):
      """
      log_probs = self.logits[i][:,None,:]
      log_probs = softmax(log_probs,-1)
      log_probs = np.log(log_probs)
      targets = self.trueSeqs[i][None,...]
      input_lengths = np.asarray([self.logitLengths[i]])
      target_lengths =np.asarray([len(targets[0])])
      loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
      """
      ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean')
      log_probs = torch.tensor(self.logits[i][:,None,:], dtype=torch.float32)  # Shape (T, N=1, C)
      targets = torch.tensor(self.trueSeqs[i], dtype=torch.int32)  # Target sequence
      input_lengths = torch.tensor([self.logitLengths[i]], dtype=torch.int32)  # Length of the input sequence
      target_lengths = torch.tensor([len(self.trueSeqs[i])], dtype=torch.int32)  # Length of the target sequence

      # Calculate the CTC loss
      loss = ctc_loss_fn(log_probs.log_softmax(-1), targets, input_lengths, target_lengths).item()
      all_ctc += loss
    return all_ctc / num_seqs

  def calculate_ctc_pday(self):
    all_cer = np.zeros(len(np.unique(self.days)))
    counts_pday = np.zeros(len(np.unique(self.days)))
    for i in range(len(self.logits)):
      day_idx = self.days[i]
      """
      log_probs = self.logits[i][:,None,:]
      log_probs = softmax(log_probs,-1)
      log_probs = np.log(log_probs)
      targets = self.trueSeqs[i][None,...]
      input_lengths = np.asarray([self.logitLengths[i]])
      target_lengths = np.asarray([len(targets[0])])
      loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
      """
      ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean')
      log_probs = torch.tensor(self.logits[i][:,None,:], dtype=torch.float32)  # Shape (T, N=1, C)
      targets = torch.tensor(self.trueSeqs[i], dtype=torch.int32)  # Target sequence
      input_lengths = torch.tensor([self.logitLengths[i]], dtype=torch.int32)  # Length of the input sequence
      target_lengths = torch.tensor([len(self.trueSeqs[i])], dtype=torch.int32)  # Length of the target sequence

      # Calculate the CTC loss
      loss = ctc_loss_fn(log_probs.log_softmax(-1), targets, input_lengths, target_lengths).item()
      all_cer[day_idx] += loss
      counts_pday[day_idx] += 1
    return all_cer / counts_pday

  def get_predictions(self, idx=None):

      all_logits = self.get_logits(idx)

      all_predictions = []
      if not(idx is None):
        all_logits = [all_logits]

      for logits in all_logits:
        # compute the softmax to get probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        # Get predicted phonemes and confidence for each time step
        predictions = np.argmax(probabilities, axis=-1)

        # remove unique and blank tokens
        predictions = unique_consecutive(predictions)
        predictions = predictions[predictions != 0]
        all_predictions.append(predictions)
      return all_predictions

  def get_predicted_phonemes(self, idx=None):

    predictions = self.get_predictions(idx)
    all_phonemes = []
    for sample in predictions:
      phonemes = ids_to_phonemes(sample)
      all_phonemes.append(phonemes)
    return all_phonemes

  def check_prediction_lengths(self):
    phonemes1 = self.get_predictions()
    true_labels1 = self.get_true_labels()
    # %
    length_differs = np.zeros(len(phonemes1))
    # Check if the classes are the same?
    for i in range(len(phonemes1)):
      l = len(phonemes1[i])
      l1 = len(true_labels1[i])
      length_differs[i] = l != l1

    print(f"Lengths differ: {int(length_differs.sum())}")
    #return length_differs






