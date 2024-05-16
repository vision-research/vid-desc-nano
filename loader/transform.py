import math
import re
import string

from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch


class UniformSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            return frames

        sample_indices = np.linspace(0, n_frames-1, self.n_sample, dtype=int)
        samples = [ frames[i] for i in sample_indices ]
        return samples


class RandomSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            return frames

        block_len = int(n_frames / self.n_sample)
        start_of_final_clip = n_frames - block_len - 1
        uniformly_sampled_indices = np.linspace(0, start_of_final_clip, self.n_sample, dtype=int)
        random_noise = np.random.choice(block_len, self.n_sample, replace=True)
        randomly_sampled_indices = uniformly_sampled_indices + random_noise
        samples = [ frames[i] for i in randomly_sampled_indices ]
        return samples


class TrimIfLongerThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        if len(frames) > self.n:
            frames = frames[:self.n]
        return frames


class ZeroPadIfLessThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        while len(frames) < self.n:
            frames = np.vstack([ frames, np.zeros_like(frames[0]) ])
        return frames


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, array):
        np_array = np.asarray(array)
        t = torch.from_numpy(np_array)
        if self.dtype:
            t = t.type(self.dtype)
        return t


class NLTKWordpunctTokenizer:

    def __call__(self, sentence):
        return wordpunct_tokenize(sentence)
# takes in a sentence as an input and returns a list of tokens (words) obtained by 
# tokenizing the sentence using the wordpunct_tokenize function



class TrimExceptAscii: # trims a sentence by removing all non-ASCII characters.
    def __init__(self, corpus):
        self.corpus = corpus

    def __call__(self, sentence):
        if self.corpus == "MSVD":
            
            s = sentence#.decode('ascii', 'ignore').encode('ascii')
        elif self.corpus == "MSR-VTT":
            s = sentence.encode('ascii', 'ignore')
        return s

# If corpus is "MSVD", the sentence is passed through without modification.
# If corpus is "MSR-VTT", the sentence is encoded with the 'ascii' encoding, and all non-ASCII characters are ignored.    


class RemovePunctuation:  # Adel
    def __init__(self):
        #self.regex = re.compile('[%s]' % re.escape(string.punctuation)) # original
        
        punctuation_chars = string.punctuation.replace('+', '') #  to keep +
        self.regex = re.compile('[%s]' % re.escape(punctuation_chars))

    def __call__(self, sentence):
        return self.regex.sub('', sentence)


class Lowercase:

    def __call__(self, sentence):
        return sentence.lower()


class SplitWithWhiteSpace:

    def __call__(self, sentence):
        return sentence.split()


class Truncate: # truncate a list of words to a specified number of words
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]


class PadFirst: # Both classes PadFirst and PadLast are used to add a token (e.g. "<SOS>", "<EOS>") 
                            # at the beginning or end of a list of words
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return [ self.token ] + words

 
class PadLast:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return words + [ self.token ]


class PadToLength: #  is a class that is used to pad a given list of words to a specified length using a specified token.
    def __init__(self, token, length):
        self.token = token
        self.length = length

    def __call__(self, words):
        n_pads = self.length - len(words)
        return words + [ self.token ] * n_pads


class ToIndex:
    def __init__(self, word2idx):
        self.word2idx = word2idx

    def __call__(self, words): # Ignore unknown (or trimmed) words.
        return [ self.word2idx[word] for word in words ]

# ToIndex
'''
The ToIndex class is a transformation that converts a list of words (strings) 
to a list of indices (integers) using a word-to-index mapping (word2idx). 
It maps each word in the input list to its corresponding index in the word-to-index mapping 
and returns the resulting list of indices. Any unknown or trimmed words are ignored during the mapping process.
'''











