# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np

def viterbi(model, observation, tags):
  """
  Returns the model's predicted tag sequence for a particular observation.
  Use `get_tag_likelihood` method to obtain model scores at each iteration.

  Input: 
    model: HMM model
    observation: List[String]
    tags: List[String]
  Output:
    predictions: List[String]
  """
  # YOUR CODE HERE 
  
  prev_probs = {}
  prev_tag_seq = {}
  curr_probs = {}
  curr_tag_seq = {}

  for i, word in enumerate(observation):
    # starting word
    if i == 0:
      for tag in tags:
        curr_probs[tag] = model.get_tag_likelihood(tag, None, observation, 0)
        curr_tag_seq[tag] = []

    else:
      for tag in tags:
        max_prob = -np.inf
        max_prev_tag = None
      
        for prev_tag in tags:
          possible_curr_prob = prev_probs[prev_tag] + model.get_tag_likelihood(tag, prev_tag, observation, i)
          if possible_curr_prob > max_prob:
            max_prev_tag = prev_tag
            max_prob = possible_curr_prob

        curr_probs[tag] = max_prob
        curr_tag_seq[tag] = prev_tag_seq[max_prev_tag] + [max_prev_tag]

    prev_probs = curr_probs
    prev_tag_seq = curr_tag_seq
    curr_probs = {}
    curr_tag_seq = {}
  
  # last token (qf)
  max_prob = -np.inf
  max_prev_tag = None

  for prev_tag in tags:
    possible_curr_prob = prev_probs[prev_tag] + model.get_tag_likelihood("qf", prev_tag, observation, len(observation))
    if possible_curr_prob > max_prob:
      max_prev_tag = prev_tag
      max_prob = possible_curr_prob

  return prev_tag_seq[max_prev_tag] + [max_prev_tag]
    

      