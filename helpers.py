# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np 
import math
from collections import Counter, defaultdict

def handle_unknown_words(t, documents): 
    """
    Replaces tokens in the given documents with <unk> (unknown) tokens if they occur 
    less frequently than a certain threshold, based on the provided parameter 't'. 
    Tokens are ordered first by frequency then alphabetically, so the tokens 
    replaced are the least frequent tokens and earliest alphabetically.
    
    Input:
        t (float):
            A value between 0 and 1 representing the threshold for token frequency.
            The int(t * total_unique_tokens) least frequent tokens will be replaced.
        documents (list of lists):
            A list of documents, where each document is represented as a list of tokens.
    Output:
        new_documents (list of lists):
            A list of processed documents where the int(t * total_unique_tokens) least
            frequent tokens have been replaced with <unk> tokens and no other changes. 
        vocab (list):
            A list of tokens representing the vocabulary, including both the most common tokens
            and the <unk> token.
    Example:
    t = 0.3
    documents = [["apple", "banana", "apple", "orange"],
                 ["apple", "cherry", "banana", "banana"],
                 ["cherry", "apple", "banana"]]
    new_documents, vocab = handle_unknown_words(t, documents)
    # new_documents:
    # [['apple', 'banana', 'apple', '<unk>'],
    #  ['apple', 'cherry', 'banana', 'banana'],
    #  ['cherry', 'apple', 'banana']]
    # vocab: ['banana', 'apple', 'cherry', '<unk>']
    """
    # YOUR CODE HERE
    
        
    all_words = sum(documents, [])

    unique, counts = np.unique(all_words, return_counts=True)
    # order first by frequency, then alphabetically
    # --> stable sort: first sort alphabetically, then by frequency

    words = list(zip(unique, counts))
    words.sort(
        key=lambda x: x[0]
    )
    words.sort(
        key=lambda x: x[1]
    )
    unique, counts = list(zip(*words))
    num_replace = int(t * len(unique))

    to_replace = unique[:num_replace]
    vocab = list(unique[num_replace:][::-1]) + ["<unk>"]

    new_documents = []
    for d in documents:
        new_doc = []
        for word in d:
            if word in to_replace:
                word = "<unk>"
            new_doc.append(word)
        new_documents.append(new_doc)

    return new_documents, vocab


def apply_smoothing(k, observation_counts, unique_obs):
    """
    Apply add-k smoothing to state-observation counts and return the log smoothed observation 
    probabilities log[P(observation | state)].

    Input:
        k (float): 
            A float number to add to each count (the k in add-k smoothing)
            Observation here can be either an NER tag or a word, 
            depending on if you are applying_smoothing to transition_matrix or emission_matrix
        observation_counts (Dict[Tuple[str, str], float]): 
            A dictionary containing observation counts for each state.
            Keys are state-observation pairs and values are numbers of occurrences of the key.
            Keys should contain  all possible combinations of (state, observation) pairs. 
            i.e. if a `(NER tag, word)` doesn't appear in the training data, you should still include it as `observation_counts[(NER tag, word)]=0`
        unique_obs (List[str]):
            A list of string containing all the unique observation in the dataset. 
            If you are applying smoothing to the transition matrix, unique_obs contains all the possible NER tags in the dataset.
            If you are applying smoothing to the emission matrix, unique_obs contains the vocabulary in the dataset

    Output:
        Dict<key Tuple[String, String]: value Float>
            A dictionary containing log smoothed observation **probabilities** for each state.
            Keys are state-observation pairs and values are the log smoothed 
            probability of occurrences of the key.
            The output should be the same size as observation_counts.

    Note that the function will be applied to both transition_matrix and emission_matrix. 
    """
    # YOUR CODE HERE 
    tags, words = list(zip(*observation_counts.keys()))
    tags, words = set(tags), set(words)
    n_tags, n_words = len(list(tags)), len(list(words))
    if set(tags) == set(unique_obs):
        index = 0
    elif set(words) == set(unique_obs):
        index = 1
    else:
        raise ValueError
    
    # use np array to store counts
    tags_words, values = zip(*observation_counts.items())

    values = np.array(values).reshape((n_tags, n_words)) + k
    values = values / values.sum(axis=index, keepdims=True)
    values = np.log(values)

    flat = values.reshape(-1,).tolist()
    smoothed_obs = {tw: v for tw, v in zip(tags_words, flat)}

    return smoothed_obs
        
