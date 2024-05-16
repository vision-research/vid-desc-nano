import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.decoder import mask


class CaptionGenerator(nn.Module):
    def __init__(self, decoder, max_caption_len, vocab, output_size ):
        super(CaptionGenerator, self).__init__()
        self.decoder = decoder
       # self.reconstructor = reconstructor
        self.max_caption_len = max_caption_len
        self.vocab = vocab
        self.output_size = output_size



    def forward_decoder(self, batch_size, vocab_size, feats, captions, masks=None ):
        
        
        outputs  = self.decoder(feats, captions, masks )
        #print('$$$$$$$$$$$$$$$$$',feats.shape)
        
        
       
        return outputs  



    def forward(self, feats, captions, masks=None):
        batch_size = feats.size(0)
        vocab_size = self.output_size
         
        outputs   = self.forward_decoder(batch_size, vocab_size, feats, captions, masks )
                                                        
      
       # print('outputs===', outputs.shape) # outputs=== torch.Size([12, 4, 9207]) where 12 = seq_len+2 , 4=bsz, 9207=vocab size
        #print('outputs===', outputs)
        if captions is None:
            _, captions = outputs.max(dim=2)
        caption_masks = (captions != self.vocab.word2idx['<PAD>']) * (captions != self.vocab.word2idx['<EOS>'])
        caption_masks = caption_masks.cuda()

            
        
       # print('outputs00===', outputs.shape) # outputs00=== torch.Size([12, 4, 9207]) # 4 = bsz , 12 = sel_len+2
       # print('outputs11===', outputs) #
        return outputs

    def describe(self, feats, beam_width, beam_alpha):
        batch_size = feats.size(0)
        vocab_size = self.output_size
     #   print('44444444444444444', vocab_size)

        captions = self.beam_search(batch_size, vocab_size, feats, beam_width, beam_alpha)
        
       # print('captions===', captions) same outputs below
        
        return captions


    def beam_search_ok00(self, batch_size, vocab_size,  feats, width, alpha):
        
      
        
       
        
        input = torch.zeros((1, self.max_caption_len + 1)).cuda()
        input = input.long()
        input[:,0] = 1
        
        
        
        for i in range(self.max_caption_len + 1): # 0 to 5 = 6
            
   #         target_pad_mask = (input == 0).cuda()
            
            masks = mask(feats[:, :, 0], input, 0)
         #   print('masks==========', masks)
         #   print('masks==========', input.dtype)
            output = self.decoder(feats, input, masks)#torch.Size([31, 1, 9207])
         #   print('$$$$$$$$$$$$', output)
          #  output_index = output.argsort(dim=2, descending=True)  
            output = output.permute(1, 0,2)
            output_index = torch.argmax(output, dim=-1)
      #      print('input===', input) 
      #      print('target_pad_mask===', target_pad_mask) 
      #      print('target_subsequent_mask===', target_subsequent_mask) 
          #  print('output===', output) 
      #      print('output===', output.shape) 
            
         
            
            token = output_index[i,:]
        #    print('token===', token.shape) 
         #   print('token===', token)
            
            
            token_int = token.item() # convert torch value to normal value not torch
            j = i + 1
            
       #     print('j===============', j) 
            
            input[:,j] = token
            
            
      #      print('token===', token) 
      #      print('input===', input) 
       #     print('@@@@@@@===', input.size(1)) 
            
            if token_int == 2  or j == (input.size(1) - 1) :
                break
            
     #   print('$$$$$$$$', input)      
      #  7/0  
        return input 
            
            
    def beam_search(self, batch_size, vocab_size,  feats, width, alpha):
        
        SOS_IDX = 1
        EOS_IDX = 2
        scores = torch.Tensor([0.])
        scores= scores.cuda()
        beam_size = width
        vocab_size = torch.tensor(vocab_size)
       
        
#        decoder_input = torch.zeros((1, self.max_caption_len + 1)).cuda()
#        decoder_input = decoder_input.long()
#        decoder_input[:,0] = 1
        
        decoder_input = torch.Tensor([[SOS_IDX]]).long()
        decoder_input = decoder_input.cuda()
        
        for i in range(self.max_caption_len + 1): # 0 to 5 = 6
            
   #         target_pad_mask = (input == 0).cuda()
            
            masks = mask(feats[:, :, 0], decoder_input, 0)
         #   print('masks==========', masks)
         #   print('masks==========', input.dtype)
            logits = self.decoder(feats, decoder_input, masks)#torch.Size([31, 1, 9207])
           # print('$$$$$$$$$$$$', logits)
          #  output_index = output.argsort(dim=2, descending=True)  
   #         logits = logits.permute(1, 0,2)

            # Softmax
            log_probs = torch.log_softmax(logits[:, -1], dim=1)
       #     print('$$$$$$$$$$$$', log_probs.shape)
       #     print('$$$$$$$$$$$$', log_probs )
            log_probs = log_probs / sequence_length_penalty(i+1, alpha)
         #   print('$$111111', log_probs.shape)
         #   print('$$$$11111111', log_probs )

            # Set score to zero where EOS has been reached
            log_probs[decoder_input[:, -1]==EOS_IDX, :] = 0
            aa=log_probs[decoder_input[:, -1]==EOS_IDX, :] = 0
           
         #   print('aaaaaaaaaa=', aa )
                    
            # scores [beam_size, 1], log_probs [beam_size, vocab_size]
            scores = scores.unsqueeze(1) + log_probs
          #  print('$$$333333333333333', scores.shape)
          #  print('$$$33333333333333333', scores )
            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, 
            # and reconstruct beam indices and token indices
            scores, indices = torch.topk(scores.reshape(-1), beam_size)
         #   print('$$$244444444444', scores.shape)
         #   print('$$$$444444444444', scores )
            
         #   print('$$$24555555555555555', indices.shape)
         #   print('$$$$44555555555555554', indices )
            beam_indices  = torch.floor_divide(indices, vocab_size ).cuda() # indices // vocab_size
            token_indices = torch.remainder(indices, vocab_size).cuda()                         # indices %  vocab_size


                # Build the next decoder input
            next_decoder_input = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_decoder_input = decoder_input[beam_index]
                if prev_decoder_input[-1]==EOS_IDX:
                    token_index = EOS_IDX # once EOS, always EOS
                token_index = torch.LongTensor([token_index]).cuda()
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
            decoder_input = torch.vstack(next_decoder_input)


            # If all beams are finished, exit
            if (decoder_input[:, -1]==EOS_IDX).sum() == beam_size:
                break

            if i==0:
                feats = feats.expand(beam_size, *feats.shape[1:])


        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        
        decoder_output = decoder_output.unsqueeze(0)




        return decoder_output 
    
    
    
    
                
    def get_cap_real_greedy00(self, feats): # greedy search
     
         
            
        input = torch.zeros((1, self.max_caption_len + 1)).cuda()
        input = input.long()
        input[:,0] = 1
            
            
            
        for i in range(self.max_caption_len + 1): # 0 to 5 = 6
                 
            masks = mask(feats[:, :, 0], input, 0)
         
            output = self.decoder(feats, input, masks)#torch.Size([31, 1, 9207])
           
            output = output.permute(1, 0,2)
            output_index = torch.argmax(output, dim=-1)
          
            token = output_index[i,:]
            
            token_int = token.item() # convert torch value to normal value not torch
            j = i + 1
                 
            input[:,j] = token
                 
            if token_int == 2  or j == (input.size(1) - 1) :
                break
         
        return input       
    
#############################################################################    
    def get_cap_real_beam00(self, feats): # beam search
        batch_size, seq_len, feat_dim = feats.shape
        vocab_size = self.output_size
        
        # Assuming you have beam_search function defined in your class
        beam_width = 5  # Set your beam width
        beam_alpha = 0.5  # Set your alpha value for length normalization

        captions = []
        for i in range(batch_size):
            feat = feats#[i:i+1, :, :]  # Take a single video's features
            caption = self.beam_search(batch_size=1, vocab_size=vocab_size, feats=feat,
                                       width=beam_width, alpha=beam_alpha)
            
            
           # captions.append(caption)

        return caption #torch.cat(captions, dim=0)


    
    
    
    
    
    
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha 



