import numpy as np
import re
import pronouncing
import heapq
import transformers
from openai import OpenAI
import os

def unique_consecutive(arr, return_indices=False):
  """Find unique consecutive elements in an array.

  Args:
      arr (numpy.ndarray): The input array.
      return_indices (bool, optional): Whether to return the indices of the unique elements. Defaults to False.

  Returns:
      numpy.ndarray or tuple: If `return_indices` is False, returns the unique consecutive elements.
                               If `return_indices` is True, returns a tuple of the unique elements and their indices.
  """

  # Find the indices where the array changes value
  change_indices = np.where(np.diff(arr) != 0)[0] + 1

  # Add 0 and the length of the array to the indices
  indices = np.concatenate(([0], change_indices, [len(arr)]))

  if return_indices:
    return arr[indices[:-1]], indices[:-1]
  else:
    return arr[indices[:-1]]



def wer(ref, hyp):
  """
  Computes the Word Error Rate (WER) between a reference and hypothesis transcription.

  Args:
  ref (str): The reference transcription.
  hyp (str): The hypothesis transcription.

  Returns:
  float: The WER score.
  """
  # Split the sentences into words
  ref_words = ref.split()
  hyp_words = hyp.split()

  # Initialize the distance matrix
  d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint8)

  # Fill in the distance matrix
  for i in range(len(ref_words) + 1):
    d[i][0] = i
  for j in range(len(hyp_words) + 1):
    d[0][j] = j

  # Compute the minimum edit distance
  for i in range(1, len(ref_words) + 1):
    for j in range(1, len(hyp_words) + 1):
      if ref_words[i - 1] == hyp_words[j - 1]:
        d[i][j] = d[i - 1][j - 1]
      else:
        substitution = d[i - 1][j - 1] + 1
        insertion = d[i][j - 1] + 1
        deletion = d[i - 1][j] + 1
        d[i][j] = min(substitution, insertion, deletion)

  # The WER is the ratio of the Levenshtein distance to the reference length
  wer_value = d[len(ref_words)][len(hyp_words)] / float(len(ref_words))
  return wer_value



client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

  # Replace with your actual OpenAI API key

# Define the phoneme set and the extended version with SIL (silence)
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']

def phoneToId(p):
  # Function to convert phoneme to its corresponding ID (index)
  return PHONE_DEF_SIL.index(p)

def remove_stress(phoneme):
  # Function to remove stress markers from phonemes
  return ''.join([char for char in phoneme if not char.isdigit()])

def sentence_to_phonemes(sentence):
  # Function to convert a sentence into phonemes using CMU Pronouncing Dictionary
  words = sentence.lower().split()
  phoneme_sequence = []

  for word in words:
    # Get the ARPAbet phonemes for the word from CMU Pronouncing Dictionary
    phonemes = pronouncing.phones_for_word(word)
    if phonemes:
      # Use the first phonetic transcription (some words have multiple pronunciations)
      phoneme_list = phonemes[0].split()
      # Remove stress markers (0, 1, 2) from phonemes
      phoneme_list = [remove_stress(phoneme) for phoneme in phoneme_list]
      phoneme_sequence.extend(phoneme_list)
    else:
      # Handle OOV (Out of Vocabulary) words (e.g., return [OOV] or some fallback method)
      phoneme_sequence.append('[OOV]')  # Placeholder for OOV words

  return phoneme_sequence

def add_noise(phoneme_ids, noise_level=0.1):
  # Function to add noise to phoneme IDs
  noisy_ids = phoneme_ids.copy()
  num_noisy_ids = int(len(phoneme_ids) * noise_level)
  for _ in range(num_noisy_ids):
      idx = np.random.randint(0, len(phoneme_ids))
      noisy_ids[idx] = np.random.randint(1, len(PHONE_DEF_SIL) + 1)  # Random phoneme ID
  return noisy_ids

def idToPhone(id):
  # Function to convert phoneme ID back to phoneme
  return PHONE_DEF_SIL[id - 1]  # -1 because we added +1 to ID earlier


# Function to convert phoneme IDs back to a phoneme sequence
def ids_to_phonemes(phoneme_ids):
    return [idToPhone(id) for id in phoneme_ids]

# Function to reconstruct sentence from phonemes (basic)
def phonemes_to_sentence(phoneme_sequence):
    sentence = ' '.join(phoneme_sequence)
    sentence = sentence.replace('[OOV]', '(unknown)')
    return sentence


def beam_search_decode(noisy_phoneme_ids, beam_width=3):
  # Function to implement beam search for reconstructing phonemes
  # Initialize the beam with an empty sequence and score 0
  beam = [([], 0)]

  # Iterate over all phoneme IDs in the sequence
  for t in range(len(noisy_phoneme_ids)):
    all_candidates = []
    # Expand each sequence in the current beam
    for seq, score in beam:
      for phoneme_id in range(1, len(PHONE_DEF_SIL) + 1):  # Consider all possible phoneme IDs
        new_seq = seq + [phoneme_id]
        # Here we compare the current phoneme_id to the noisy_phoneme_ids[t]
        # Penalize it if it's not the same as the original phoneme_id at timestep t
        new_score = score - (1 if phoneme_id != noisy_phoneme_ids[t] else 0)
        all_candidates.append((new_seq, new_score))

    # Select the top k sequences with the highest scores
    beam = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[1])

  # Return the sequence with the highest score
  best_sequence = beam[0][0]
  return best_sequence

# Step 3: Function to use ChatGPT to refine the sentence
def refine_sentence_with_llm(sentence):
    gpt_model = transformers.pipeline("text-generation",
                                      model="meta-llama/Llama-3.2-11B-Vision-Instruct")
    refined_sentence = gpt_model(f"Correct and refine this sentence: '{sentence}'. Your output should only be the refined sentence.", max_length=50, num_return_sequences=1)[0]['generated_text']
    return refined_sentence


# Function to use OpenAI GPT-3.5/4.0 to refine the sentence
def refine_sentence_with_chatgpt(sentence):
    # Replace with your actual OpenAI API key

  # Use OpenAI's GPT models to refine the sentence
  response = client.chat.completions.create(model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo" based on your needs
  messages=[
    {"role": "system", "content": "You are a helpful assistant that refines and corrects sentences."},
    {"role": "user",
     "content": f"Correct and refine this sentence: '{sentence}'. Your output should only be the refined sentence."}
  ])

  # Extract the refined sentence from the response
  refined_sentence = response.choices[0].message.content.strip()
  return refined_sentence

def refine_phonetic_transcript(sentence):

  # Replace with your actual OpenAI API key

  # Use OpenAI's GPT models to refine the sentence
  response = client.chat.completions.create(model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo" based on your needs
                                            messages=[
                                              {"role": "system",
                                               "content": "You are a helpful assistant that converts phonetic transcripts to sentences."},
                                              {"role": "user",
                                               "content": f"Convert the phonetic transcript to a sentence and refine the sentence: '{sentence}'. Your output should only be the refined sentence."}
                                            ])

  # Extract the refined sentence from the response
  refined_sentence = response.choices[0].message.content.strip()
  return refined_sentence


if __name__ == '__main__':

  # Example sentence
  sentence = "my mother is a teacher"

  # Convert the sentence to phonemes
  phonemes = sentence_to_phonemes(sentence)

  # Convert the phoneme to its corresponding ID
  phoneme_ids = [phoneToId(p) + 1 for p in phonemes]

  #Add noise
  noisy_phoneme_ids = add_noise(phoneme_ids, noise_level=0.2)
  print(f"Noisy Phoneme IDs: {noisy_phoneme_ids}")

  # Beam search decoding to find the best sequence of phoneme IDs
  best_phoneme_ids = beam_search_decode(noisy_phoneme_ids, beam_width=3)
  print(f"Best Phoneme IDs (Beam Search): {best_phoneme_ids}")

  # Convert the best phoneme IDs back to phonemes
  best_phonemes = ids_to_phonemes(best_phoneme_ids)
  print(f"Best Phonemes: {best_phonemes}")

  # Reconstruct the sentence from the best phonemes
  reconstructed_sentence = phonemes_to_sentence(best_phonemes)
  print(f"Reconstructed Sentence: {reconstructed_sentence}")

  # Refine the sentence using ChatGPT
  refined_sentence = refine_sentence_with_chatgpt(reconstructed_sentence)

  print(f"Refined Sentence: {refined_sentence}")

  wer_score = wer(sentence, refined_sentence)
  print(f"WER Score: {wer_score}")
