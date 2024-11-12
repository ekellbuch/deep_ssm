"""
Compare logits
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn

from deep_ssm.utils import bci_utils
import seaborn as sns
from deep_ssm.utils.eval_utils import ModelOutput, count_classes
import numpy as np


def calculate_phoneme_accuracy(logits, targets, adjusted_lens):
  total_phonemes = 0
  correct_phonemes = 0

  # Iterate over each sequence in logits and targets
  for iter_idx in range(len(logits)):
    # Extract logits for the current sequence and apply softmax to get probabilities
    logit_seq = logits[iter_idx][:adjusted_lens[iter_idx], :]  # Shape (T, C)

    # Decode predicted sequence by taking the argmax (predicted class) along class dimension
    decoded_seq = np.argmax(logit_seq, axis=-1)  # Shape (T,)

    # Remove consecutive duplicates and zeros (padding)
    decoded_seq = np.array(
      [decoded_seq[i] for i in range(len(decoded_seq)) if i == 0 or decoded_seq[i] != decoded_seq[i - 1]])
    decoded_seq = decoded_seq[decoded_seq != 0]

    # Get the true sequence for this batch element
    true_seq = targets[iter_idx]  # Shape (S,)

    # Ensure we're comparing up to the length of the true sequence
    total_phonemes += len(true_seq)
    correct_phonemes += sum(decoded_seq[:len(true_seq)] == true_seq)

  # Return the phoneme-level accuracy
  return correct_phonemes / total_phonemes


# Extend the KL divergence comparison with more debugging
def detailed_kl_analysis(logits_1, logits_2):
  kl_divergence_per_layer = []

  for i, (logit_1, logit_2) in enumerate(zip(logits_1, logits_2)):
    # KL Divergence over the sequence
    log_probs_1 = F.log_softmax(torch.tensor(logit_1), dim=-1)
    probs_2 = F.softmax(torch.tensor(logit_2), dim=-1)

    # Compute KL Divergence for the whole sequence
    kl_div = F.kl_div(log_probs_1, probs_2, reduction='batchmean').item()
    kl_divergence_per_layer.append(kl_div)

  print("KL Divergence per sequence:", kl_divergence_per_layer)
  return kl_divergence_per_layer


# Calculate KL Divergence per day
def calculate_kl_per_day(logits_1, logits_2, partition_name):
  print(f"Calculating KL Divergence per day for {partition_name} partition:")

  kl_per_day = {}
  for i, (logit_1, logit_2) in enumerate(zip(logits_1["logits"], logits_2["logits"])):
    day = logits_1["transcriptions"][i]

    # Convert logits to log probabilities
    log_probs_1 = F.log_softmax(torch.tensor(logit_1), dim=-1)
    probs_2 = F.softmax(torch.tensor(logit_2), dim=-1)

    # Compute KL Divergence (using batchmean reduction)
    breakpoint()
    kl_div = F.kl_div(log_probs_1, probs_2, reduction='batchmean').item()

    if day not in kl_per_day:
      kl_per_day[day] = []
    kl_per_day[day].append(kl_div)

  # Averaging KL divergence per day
  kl_per_day_avg = {day: np.mean(kl_list) for day, kl_list in kl_per_day.items()}

  return kl_per_day_avg




#%%
def main():
  #%%
  # Define the paths to the output files from the experiments
  exper1 = 'ipru93g8'
  exper2 = '71czcr82'
  partition = 'train' # 'train' or 'test'

  # Define models
  experiment_id = {"ipru93g8": "mamba",
                   "71czcr82": "gru" }
  experiment_1 = f"bci_outputs/{exper1}_model_prediction_{partition}.pkl"
  experiment_2 = f"bci_outputs/{exper2}_model_prediction_{partition}.pkl"
  # Compare in which case the CER is lower:
  # Plot logits distribution:
  label1 = "Experiment {}:{} ({})".format(exper1,experiment_id[exper1], partition)
  label2 = "Experiment {}:{} ({})".format(exper2,experiment_id[exper2], partition)


  # Load model data:
  model1 = ModelOutput(experiment_1)
  model2 = ModelOutput(experiment_2)

  # Check the length of the predictions
  model1.check_prediction_lengths()
  model2.check_prediction_lengths()

  #%%
  # Compare the length of the predictions
  phonemes1 = model1.get_predictions()
  true_labels1 = model2.get_predictions()
  # check the length of the predictions are the same:
  length_differs = np.zeros(len(phonemes1))
  # Check if the classes are the same?
  for i in range(len(phonemes1)):
    l = len(phonemes1[i])
    l1 = len(true_labels1[i])
    length_differs[i] = l != l1
  print(length_differs.sum())

  # Plot lengths of the predictions
  l1 = [len(phonemes1[i]) for i in range(len(phonemes1))]
  l2 = [len(true_labels1[i]) for i in range(len(true_labels1))]

  #%%
  plt.hist(l1, bins=50, label=label1, edgecolor='black', alpha=0.6)
  plt.hist(l2, bins=50, label=label2, edgecolor='black', alpha=0.6)
  plt.legend()
  plt.xlabel('Predicted Length')
  plt.ylabel('Count')
  plt.title('Comparison of Predicted Lengths Between Two Models')
  plt.tight_layout()
  plt.show()

  #%% Compare the class counts
  class_1 = count_classes(phonemes1)
  class_2 =  count_classes(true_labels1)
  #%%
  # character error rate:

  #%%
  # word error rate:
  #for i in range(10):
    #bci_utils.wer(phonemes1[i], true_labels1[i])

  #%%
  # Plot the class counts
  plt.bar(class_1.keys(), class_1.values(),label=label1, alpha=1)
  plt.bar(class_2.keys(), class_2.values(), label=label2, alpha=0.4)
  plt.xlabel('Class ID')
  plt.ylabel('Count')
  plt.title('Comparison of Class Counts Between Two Models')
  plt.legend()
  plt.tight_layout()
  plt.show()

#%%
  # Compare the sentences:
  model1.get_predicted_phonemes(0)

  #%%


  #%%

  """
  cer1_full = model1.calculate_metric(metric)
  cer2_full = model2.calculate_metric(metric)

  print(cer1_full, cer2_full)
  # calculate CER per day for both experiments
  cer1 = model1.calculate_metric_pday(metric)
  cer2 = model2.calculate_metric_pday(metric)

  # Plot CER per day for both experiments
  l
  plt.plot(np.arange(len(cer1)), cer1, label="Experiment {}:{} ({})".format(exper1,experiment_id[exper1], partition))
  plt.plot(np.arange(len(cer2)), cer2, label="Experiment {}:{} ({})".format(exper2,experiment_id[exper2], partition))
  plt.xlabel("Day")
  plt.ylabel(metric)
  plt.legend()
  plt.tight_layout()
  plt.show()
 
  # how are they different?
  # Perform phoneme accuracy comparison
  breakpoint()
  phoneme_acc1 = calculate_phoneme_accuracy(model1.logits, model1.trueSeqs, model1.logitLengths)
  phoneme_acc2 = calculate_phoneme_accuracy(model2.logits, model2.trueSeqs, model2.logitLengths)
  print(f"Phoneme Accuracy: Model 1 = {phoneme_acc1}, Model 2 = {phoneme_acc2}")
   """
  # Do we even have the same lengths?
  # Check if they are all the same length? yep, always th

  #%% Our issue is that
  # KL Divergence per sequence
  kl_val = detailed_kl_analysis(model1.logits, model2.logits)
  max_diff_indices =  np.argsort(kl_val)[::-1]

  #logits_1 = model1.logits[max_diff_idx]  # T x C
  #logits_2 = model2.logits[max_diff_idx]  # T x C

  # ignore first or last


  #%
  # apply decoder utils

  #%%
  #%%
  max_diff_idx = max_diff_indices[0]
  batch_1 = model1.get_sample(max_diff_idx)
  batch_2 = model2.get_sample(max_diff_idx)

  #%%
  # compare output length and true sequence length:
  for sample in range(len(model1.logits)):
    logit_len = model1.get_sample(sample)[1]
    true_len = len(model1.get_sample(sample)[2])
    breakpoint()
    print(sample, logit_len - true_len)
  #%%
  logits_1 = batch_1[0]
  logits_2 = batch_2[0]
  # vanilla - back to phoneme:
  logits = logits_1
  probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
  # Get predicted phonemes and confidence for each time step
  predictions = np.argmax(probabilities, axis=-1)
  # remove unique and blank tokens
  predictions = bci_utils.unique_consecutive(predictions)
  predictions = predictions[predictions != 0]

  true_sentence = batch_1[-2]
  trueSeqs = batch_1[-3]
  print('true sequence:', true_sentence)
  best_phonemes_original = bci_utils.ids_to_phonemes(trueSeqs)
  print('true phoneme sequence:', best_phonemes_original)
  model_phonemes_original = bci_utils.ids_to_phonemes(predictions)
  print('model phoneme sequence:', model_phonemes_original)

  #%%
  # Determine the min and max values across both logits for consistent scaling
  vmin = min(logits_1.min(), logits_2.min())
  vmax = max(logits_1.max(), logits_2.max())

  fig, axarr = plt.subplots(1, 2, figsize=(15, 5))

  # Display the logits as images with the same color scale
  im1 = axarr[0].imshow(logits_1.T, aspect='auto', vmin=vmin, vmax=vmax)
  axarr[0].set_title(label1)
  axarr[0].set_ylabel("Class ID")


  im2 = axarr[1].imshow(logits_2.T, aspect='auto', vmin=vmin, vmax=vmax)
  axarr[1].set_title(label2)

  # Add a single shared colorbar that represents the range of both models' values
  cbar = fig.colorbar(im1, ax=axarr, orientation='vertical', fraction=0.03, pad=0.02, shrink=0.8)
  cbar.set_label('Logit Values')

  # Add shared x-axis label
  fig.supxlabel("Sequence Index")

  plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the layout to fit the colorbar
  plt.show()



if __name__ == "__main__":
  main()
