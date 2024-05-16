import pandas as pd

from loader.data_loader import CustomVocab, CustomDataset, Corpus
 


deleted_words = []
del_count = 10

 

class MSVDVocab(CustomVocab):
    """ MSVD Vocaburary """

    
      
    def load_captions(self):
        captions = []
        with open(self.caption_fpath, 'r') as txt_file:
            for line in txt_file:
                
                caption = line.strip().split(' ', 1)[1]  # Extracting the caption after the first space
               
            
                captions.append(caption)
        return captions
    
     
    def load_deleted_words(self):
        global deleted_words
        deleted_words = [word for word, freq in list(self.word_freq_dict.items()) if freq < self.min_count] # < del_count

            
        

    def build(self):
        captions = self.load_captions()
        
        for caption in captions:
   #         print('qqqqqqqqqqqqqqqqqqqqqqqq', caption)# ال+ سنجاب يأكل ال+ فول ال+ سوداني في قشر +ت +ه
            
           # 7/0
            words = self.transform(caption)
     #       print('wwwwwwwwwwwwwwwwwwwwwwwwwww', words)# ['ال+', 'سنجاب', 'يأكل', 'ال+', 'فول', 'ال+', 'سوداني', 'في', 'قشر', '+ت', '+ه']
             
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [ word for word, freq in list(self.word_freq_dict.items()) if freq >= self.min_count ]
        
        self.load_deleted_words()
        
        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])
        print(' ----  vocabulary is done  ----')
        print('n_vocabs ======',self.n_vocabs)
    
      #  7/0
      # adel :
#'''          
        vocab_freq = {word: self.word_freq_dict[word] for word in self.word2idx}
        sorted_vocab_freq = sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True)


        with open('vocab_msvd.txt', 'w', encoding='utf-8') as file:
            count = 0
            for word, freq in sorted_vocab_freq:
        #        print(f"{word}: {freq}", end=", ")
                file.write(word + '\n')
                count += 1
             #   if count == 4000000:
              #       7/0
                
        num_elements = len(vocab_freq)
        print('number of el ======',num_elements)
         
#'''  
  
 
class MSVDDataset(CustomDataset):
    """ MSVD Dataset """

    
             


    def load_captions(self): #  using delted_word to remove caption that has deleted words 
        captions = []
        count = 0
        with open(self.caption_fpath, 'r') as txt_file:
            for line in txt_file:
                
                if ( self.phase == 'train' ): #  or self.phase == 'val' 
                    
                    video_id, caption = line.strip().split(' ', 1)  # Extracting the video ID and caption
                    
           
                    cap_len = len(caption.split())
                  
                     
           
               
                    if cap_len < 2: # 2
                         continue
                    else:
                         count += 1
                          
                         self.captions[video_id].append(caption)
                
                else:
                    video_id, caption = line.strip().split(' ', 1)  # Extracting the video ID and caption
                    
               #     caption = caption.split()  # Split the text into words using whitespace as the separator
                #    caption = ' '.join(caption[::-1])  # Reverse the list of words and join them back with a space separator
                    self.captions[video_id].append(caption)
                    count += 1
                    
            print('ggggggggggggggggggggggggggggggggg', count)      

               

class MSVD(Corpus): # defines a class called "MSVD" which inherits from the "Corpus" class
    """ MSVD Corpus """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDVocab, MSVDDataset)

 




