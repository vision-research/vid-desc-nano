

from collections import defaultdict # it is dict return default value if key is not exsist
                # defaultdict : a dictionary used to avoid raising KeyError when a key is not found in the dictionary.
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import torch.nn.functional as F
from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
                             RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
                             ToIndex



class CustomVocab(object): # similar to Tokenuzer in TensorFlow
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split):
        self.caption_fpath = caption_fpath
        self.min_count = min_count # minimum count of words to be considered in the vocabulary
        self.transform = transform # a function to be used to process the captions before building the vocabulary

        self.word2idx = defaultdict(lambda: init_word2idx['<UNK>']) # a dictionary maps words to their corresponding indices
       
        # when a word that is not in the word2idx dictionary is encountered, it will be mapped to the *index* of the <UNK> token.
        # '<UNK>' is a key not value of dictionary
        self.word2idx.update(init_word2idx)
        '''The name of dictionary is self.word2idx or is init_word2idx ? The name of the dictionary that is being initialized with a default value of init_word2idx['<UNK>'] is self.word2idx.
init_word2idx is an argument passed to the __init__ method when an object of the CustomVocab class is created.
self.word2idx is an instance variable of the CustomVocab class that is being initialized to a defaultdict. This defaultdict will be used to store the mapping of words to their corresponding indices.
It is updated with init_word2idx using self.word2idx.update(init_word2idx)
In this way, init_word2idx is used to initialize the self.word2idx dictionary and also to provide the default value for it.'''

        self.idx2word = { v: k for k, v in list(self.word2idx.items()) }
        self.word_freq_dict = defaultdict(lambda: 0) # is also created as a defaultdict with default value 0
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1 # initially set to -1, which is a flag value indicating that the maximum length has not been set yet.

        self.build()

# "Vocabularies" refers to the set of unique words that make up the vocabulary. In other words, it is the number of distinct words in the vocabulary.

# "Words" refers to the total number of words, including repetitions. It is the sum of the frequency of all the words in the vocabulary.

    def load_captions(self): 
        raise NotImplementedError("You should implement this function.") # build by user later based on dataset files format, json or txt ...

    def build(self):
        captions = self.load_captions()
         
        for caption in captions:
            
            words = self.transform(caption) # iterates through the captions, using the transform function to split each caption into a list of words.
            self.max_sentence_len = max(self.max_sentence_len, len(words)) # For each caption, it updates the maximum sentence length by comparing it to the length of the current caption's words.
            for word in words:
                self.word_freq_dict[word] += 1 # It also updates the word frequency dictionary self.word_freq_dict, counting the number of occurrences of each word.
       
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))
 # It records the number of vocabularies and words before trimming down the vocabulary based on the min_count argument.


        keep_words = [ word for word, freq in list(self.word_freq_dict.items()) if freq >= self.min_count ]
        # It then creates a list of words that occur at least min_count times, called keep_words.
        
        
        for idx, word in enumerate(keep_words, len(self.word2idx)): # It iterates through keep_words, updating the self.word2idx and self.idx2word dictionaries.
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])
# It updates the number of vocabularies and words after trimming down the vocabulary based on the min_count argument.

class CustomDataset(Dataset): # defining subclass of the built-in *Dataset* class from the PyTorch library
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_frame=None, transform_caption=None):
        self.C = C # a configuration object
        self.phase = phase # a string indicating the phase (e.g. "train", "val", "test")
        self.caption_fpath = caption_fpath # path to a file containing captions
        self.transform_frame = transform_frame # a function to be applied to each frame feature

        self.transform_caption = transform_caption #  a function to be applied to each caption

        self.video_feats = defaultdict(lambda: []) # defaultdict with default value [] that will store the video features.
        self.captions = defaultdict(lambda: []) # defaultdict with default value [] that will store the captions.
        self.data = [] # list that will store the video-caption pairs.
 
        self.build_video_caption_pairs() # method to build the video-caption pairs and store them in self.data.
 
    
 
# __len__ and __getitem__ that are required by the PyTorch Dataset class to enable indexing and slicing of the data.
 
    def __len__(self):
       # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ', len(self.data)) # 48779 & 4291 &
        return len(self.data)
# The __len__ method returns the length of the data.

    def __getitem__(self, idx):
        vid, video_feats, caption = self.data[idx]
        
# The __getitem__ method takes an index as input, returns the video-caption pair at that index, 
# and applies any specified transforms to the frame features and captions before returning them.
 
        if self.transform_frame:
            video_feats = [ self.transform_frame(feat) for feat in video_feats ]
        if self.transform_caption:
           
            caption = self.transform_caption(caption)
            
            
        return vid, video_feats, caption

    def load_video_feats(self): # C.feat.models // feat is instance of class *FeatureConfig* in train_stage1.py and to makes it clear that the models variable is related to feature extraction
        for model in self.C.feat.models:  
            
            if self.phase == 'train':
                fpath1 = self.C.loader.phase_video_feat_fpath_tpl_train_2d
                fpath2 = self.C.loader.phase_video_feat_fpath_tpl_train_3d
                
            if self.phase == 'val':
                fpath1 = self.C.loader.phase_video_feat_fpath_tpl_val_2d
                fpath2 = self.C.loader.phase_video_feat_fpath_tpl_val_3d
                
            if self.phase == 'test':
                fpath1 = self.C.loader.phase_video_feat_fpath_tpl_test_2d
                fpath2 = self.C.loader.phase_video_feat_fpath_tpl_test_3d
 
    
            fin1 = h5py.File(fpath1, 'r')  
            fin2 = h5py.File(fpath2, 'r')
            
            for vid in list(fin1.keys()):
                feats1 = fin1[vid][:]  
                feats2 = fin2[vid][:]  # -> (1, 768, 8, 7, 7)
      
                sampled_idxs1 = np.linspace(0, len(feats1) - 1, self.C.loader.frame_sample_len, dtype=int)
                feats1 = feats1[sampled_idxs1]
                #print('zzzzzzzzzz',feats1.shape)
      
            #    feats2 = feats2.squeeze(0) 
              
              #  feats2 = feats2.mean(dim=[2, 3, 4])
              #  feats2 = feats2.mean(dim=(2, 3, 4))
            #    feats2 = np.mean(feats2, axis=(2, 3, 4))
              
                feats2 = feats2[:, np.newaxis] 
                feats2 = np.repeat(feats2, self.C.loader.frame_sample_len, axis=1)
                
               # feats = np.concatenate((feats1, feats2), axis=-1)
                feats = np.concatenate((feats1, feats2[0]), axis=1)
                
                self.video_feats[vid].append(feats)
                
            fin1.close()
# Append the sampled features to the self.video_feats dictionary with the video id as the key.


    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_video_feats()
        self.load_captions()
        count = 0
        for vid in list(self.video_feats.keys()): # iterates through all the videos in the self.video_feats dictionary
            video_feats = self.video_feats[vid]
            
            for caption in self.captions[vid]:
                self.data.append(( vid, video_feats, caption ))
                
                  
                
               # print('caption===', self.data)
              #  print('caption===', caption)
     #   adds a tuple of the video ID, video features, and caption to the self.data list           
                
         

class Corpus(object):
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

# vocab_cls and dataset_cls are two other classes, which are used to create the vocabulary and dataset objects.

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loaders()
# In the build method, the class first creates the vocabulary using the vocab_cls class, 
# and then creates the train, validation, and test datasets using the dataset_cls class.


    def build_vocab(self):
        self.vocab = self.CustomVocab(
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<SOS>']),
            PadLast(self.vocab.word2idx['<EOS>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2), # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)

    def build_dataset(self, phase, caption_fpath):
         dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption)
         return dataset


    def collate_fn(self, batch):
        
        vids, video_feats, captions = list(zip(*batch))
        video_feats_list = list(zip(*video_feats))
        #print('111111111111111111111',captions )
        video_feats_list = [ torch.stack(video_feats) for video_feats in video_feats_list ]
        captions = torch.stack(captions)

        video_feats_list = [ video_feats.float() for video_feats in video_feats_list ]
        captions = captions.float()
      
        """ (batch, seq, feat) -> (seq, batch, feat) """
        captions = captions.transpose(0, 1)
         
        return vids, video_feats_list, captions
    
    def set_random_seed(self, seed):
        torch.manual_seed(seed)

   

    def build_data_loader(self, dataset):
        
        self.set_random_seed(seed=20)
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False, # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.collate_fn)
        return data_loader

