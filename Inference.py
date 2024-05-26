from Yolov7.detect import detect
import argparse 

import importlib
######################################################################################
# The implementation relies on http://nlp.seas.harvard.edu/2018/04/03/attention.html #
######################################################################################

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



inp_feat = 2304 # swin_s + Eff_B3
 

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class VocabularyEmbedder(nn.Module):
    
    def __init__(self, voc_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model)
        
    def forward(self, x): # x - tokens (B, seq_len)
        
    
        x = self.embedder(x)
    
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)
    
class FeatureEmbedder(nn.Module):
    
    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        d_feat = inp_feat
 
       
        self.embedder = nn.Sequential(           
            nn.Linear(d_feat, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model))
            
            
            
    
    
    def forward(self, x):  # x - tokens (B, seq_len, d_feat)
     
        batch_size, seq_len, d_feat = x.size()
    
        x = x.reshape(batch_size * seq_len, d_feat)

        x = self.embedder(x)
        x = x.view(batch_size, seq_len, self.d_model)
 
        return x  # (B, seq_len, d_model)
    
    
 

class PositionalEncoder(nn.Module):
    def __init__(self, pos_enc_mat, dout_p):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dout_p)
        self.pos_enc_mat = pos_enc_mat

    def forward(self, x):
        B, S, d_model = x.shape
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        return x

def compute_positional_encoding_matrix(seq_len, d_model):
    pos_enc_mat = np.zeros((seq_len, d_model))
    odds = np.arange(0, d_model, 2)
    evens = np.arange(1, d_model, 2)

    for pos in range(seq_len):
        pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
        pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))
    
    return torch.from_numpy(pos_enc_mat).unsqueeze(0)


 

class PositionalEncoder_feat(nn.Module):
    
    def __init__(self, d_model, dout_p, seq_len=1536): # 3651 max feat len for c3d
        super(PositionalEncoder_feat, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))
        
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)
        
    def forward(self, x): # x - embeddings (B, seq_len, d_model)
        B, S, d_model = x.shape
    
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        
        return x # same as input


def subsequent_mask(size):
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)
    
    return mask.byte() # ([1, size, size])

 

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def attention(Q, K, V, mask):
 
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)
    
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    
    return out # (B, *(H), seq_len, d_model//H = d_k)

class MultiheadedAttention(nn.Module):
    
    def __init__(self, d_model, H):
        super(MultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        # Q, K, V, and after-attention layer (4). out_features is d_model
        # because we apply linear at all heads at the same time
        self.linears = clone(nn.Linear(d_model, d_model), 4) # bias True??
        
    def forward(self, Q, K, V, mask): # Q, K, V are of size (B, seq_len, d_model)
  #      print('xxxxxxxxxxxxxxxxxxxxx',Q.shape)
        B, seq_len, d_model = Q.shape
        
        Q = self.linears[0](Q) # (B, *, in_features) -> (B, *, out_features)
        K = self.linears[1](K)
        V = self.linears[2](V)
        
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2) # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1)
        
        # todo: check whether they are both calculated here and how can be 
        # serve both.
        att = attention(Q, K, V, mask) # (B, H, seq_len, d_k)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
        att = self.linears[3](att)
        
        return att # (B, H, seq_len, d_k)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)
        
    def forward(self, x, sublayer): # [(B, seq_len, d_model), attention or feed forward]
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        
        return x + res
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # todo dropout?
        
    def forward(self, x): # x - (B, seq_len, d_model)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x # x - (B, seq_len, d_model)
    
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, None)
        
        sublayer1 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        
        x = self.res_layers[1](x, sublayer1)
        
        return x # x - (B, seq_len, d_model)
    
class Encoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
         
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        for layer in self.enc_layers:
            x = layer(x, None)
        
        return x # x - (B, seq_len, d_model) which will be used as Q and K in decoder
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.self_att = MultiheadedAttention(d_model, H)
        self.enc_att =  MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
       
        
    def forward(self, x, memory, src_mask, trg_mask): # x, memory - (B, seq_len, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        # a comment regarding the motivation of the lambda function 
        # please see the EncoderLayer
        sublayer0 = lambda x: self.self_att(x, x, x, trg_mask)
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
    #    print('using %%%%%%%%%%%%%%%%%%%%%%%',trg_mask.shape)
        sublayer2 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        
        return x # x, memory - (B, seq_len, d_model)
    
class Decoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
         
        self.dec_layers = clone(DecoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, memory, src_mask, trg_mask): # x (B, S, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)
        # todo: norm?
        return x # (B, S, d_model)
    
class SubsAudioVideoGeneratorConcatLinearDoutLinear(nn.Module):
    
    def __init__(self, d_model_video, voc_size, dout_p):
        super(SubsAudioVideoGeneratorConcatLinearDoutLinear, self).__init__()
        self.linear = nn.Linear( d_model_video, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        print('using SubsAudioVideoGeneratorConcatLinearDoutLinear')
       
        
    # ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_video)
    def forward(self, video_x):
        x = video_x #torch.cat([subs_x, audio_x, video_x], dim=-1)
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))  # original
   
        
        return F.log_softmax(x, dim=-1) # (B, seq_len, voc_size)


class Transformer(nn.Module):
    
    def __init__(self, output_size,
                 d_feat_video,
                 d_model_video,
                 d_ff_video,
                 N_video,
                 dout_p, H, use_linear_embedder, use_encoder):
        super(Transformer, self).__init__()
        
        if use_linear_embedder:
    
            self.src_emb_video = FeatureEmbedder(d_feat_video, d_model_video)
        else:
            assert d_feat_video == d_model_video 
       
            self.src_emb_video = Identity()
        
        trg_voc_size = output_size
      
       
        self.trg_emb_video = VocabularyEmbedder(trg_voc_size, d_model_video)
     
        self.pos_emb_video_feat = PositionalEncoder_feat(d_model_video, dout_p)
   
        # Example usage:
        seq_len = 16  #   sequence length = 15 + <SOS>
        d_model = 1024  #   model dimension
        pos_enc_mat = compute_positional_encoding_matrix(seq_len, d_model)

        # Instantiate PositionalEncoder with the precomputed positional encoding matrix
        self.pos_emb_video = PositionalEncoder(pos_enc_mat, dout_p=0.1)


 
        if use_encoder:
            self.encoder_video = Encoder(d_model_video, dout_p, H, d_ff_video, N_video)
        else:
            self.encoder_video = Identity()
     
        
     
        self.decoder_video = Decoder(d_model_video, dout_p, H, d_ff_video, N_video)
        
        # late fusion
        self.generator = SubsAudioVideoGeneratorConcatLinearDoutLinear(
            d_model_video, trg_voc_size, dout_p
        )
        
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
 
    def forward(self, src, trg, mask ):
  
        trg_mask = mask
           
        src_mask = None
           
        src_video = src
      
        src_video = self.src_emb_video(src_video)
 
        trg_video = self.trg_emb_video(trg)
        trg_video = self.pos_emb_video(trg_video)
        memory_video = self.encoder_video(src_video  ) # (src_video) only if encoder False
        out_video = self.decoder_video(trg_video, memory_video, src_mask, trg_mask)
 
        
        out = self.generator( out_video)
        
 
        return out # (B, St, voc_size)

 
############################################################################################################## 
#############################################################################################################
################################################################################################################
################################################################################################################## 
 
 
 
import time 
 
import torch.nn as nn
 
class CaptionGenerator(nn.Module):
    def __init__(self, decoder, max_caption_len, vocab, output_size ):
        super(CaptionGenerator, self).__init__()
        self.decoder = decoder
       # self.reconstructor = reconstructor
        self.max_caption_len = max_caption_len
        self.vocab = vocab
        self.output_size = output_size


 

    def describe(self, scores, decoder_input, beam_indices, token_indices,Token_index,feats, beam_width, beam_alpha):
        batch_size = feats.size(0)
        vocab_size = self.output_size
    
        
        captions = self.beam_search(scores, decoder_input, beam_indices, token_indices,Token_index,batch_size, vocab_size, feats, beam_width, beam_alpha)
        
      
        
        return captions


   
            
    def beam_search(self, scores, decoder_input, beam_indices, token_indices,Token_index, batch_size, vocab_size,  feats, width, alpha):
        
      
        EOS_IDX = 2
      
        beam_size = width
        vocab_size = torch.tensor(vocab_size)
       

        
        for i in range(16):   # max sequence + <SOS> token
             
            masks = None
  
            logits = self.decoder(feats, decoder_input, masks) 
 
        
            log_probs = torch.log_softmax(logits[:, -1], dim=1)
            log_probs = log_probs / sequence_length_penalty(i+1, alpha)
          

            # Set score to zero where EOS has been reached
            log_probs[decoder_input[:, -1]==EOS_IDX, :] = 0
            
                    
            # scores [beam_size, 1], log_probs [beam_size, vocab_size]
            scores = scores.unsqueeze(1) + log_probs

            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, 
            # and reconstruct beam indices and token indices
 
            scores, indices = torch.topk(scores.reshape(-1), beam_size)
             
            beam_indices  = torch.floor_divide(indices, vocab_size ) 
            token_indices = torch.remainder(indices, vocab_size)                          


                # Build the next decoder input
            next_decoder_input = []

   
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_decoder_input = decoder_input[beam_index] 
     
                if prev_decoder_input[-1]==EOS_IDX:
                    token_index = EOS_IDX  
                Token_index = torch.LongTensor([token_index]).cuda()
              
                next_decoder_input.append(torch.cat([prev_decoder_input, Token_index]))

 
            decoder_input = torch.vstack(next_decoder_input)
 
            # If all beams are finished, exit
            if (decoder_input[:, -1]==EOS_IDX).sum() == beam_size:
                break

            if i==0:
                feats = feats.expand(beam_size, *feats.shape[1:])




        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        
        decoder_output = decoder_output.unsqueeze(0)
 
        return decoder_output 
    
     
    
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha 




############################################################################################################## 
#############################################################################################################
################################################################################################################
################################################################################################################## 
 
 
import cv2
 
import torch.nn as nn
 
from mmcv import Config  
from Video_Swin_Transformer.mmaction.models import build_model
from mmcv.runner import  load_checkpoint
 
alpha = 0.5
BEAM = 3

   
# Load pre-trained EfficientNet-B3 model
from timm import create_model

model1 = create_model('efficientnet_b3', pretrained=True)

 
model1 = nn.Sequential(*list(model1.children())[:-1])
model1.cuda()
model1.eval() 
 
      
########################################################################################################
  
########################################################################################################
  
            ## Swin model
            
  
config2 = './Video_Swin_Transformer/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py'
checkpoint = './Video_Swin_Transformer/checkpoints/swin_small_patch244_window877_kinetics400_1k.pth'

cfg2 = Config.fromfile(config2)
model2 = build_model(cfg2.model, train_cfg=None, test_cfg=cfg2.get('test_cfg'))
 
model2.eval() 
model2.cuda()
 # Load the checkpoint onto the GPU
checkpoint = load_checkpoint(model2, checkpoint, map_location='cuda')
    
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)   # Add crop also   
  
def center_crop_frame(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx, :]
 
 
def video_description(model, vocab, beam_width=BEAM, beam_alpha=alpha):
    model.eval()
 
    cap = cv2.VideoCapture(0) # '/dev/video0'
    if cap.isOpened():
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
      
    EOS_idx = vocab.word2idx['<EOS>'] 
    count = 0
    
    c = 0
    
   
    frame_list1 = []
    frame_list2 = []
    
    last_6_frame = []
    


########################################################################################################

    scores = torch.Tensor([0.])
    scores= scores.cuda()

    SOS_IDX = 1
    decoder_input = torch.Tensor([[SOS_IDX]]).long()
    decoder_input = decoder_input.cuda()
   
   
    indices = torch.tensor([1, 1, 1]) # like beam size
    vocab_size = torch.tensor(2831)   # vocab_size

    beam_indices  = torch.floor_divide(indices, vocab_size ).cuda()  
    token_indices = torch.remainder(indices, vocab_size).cuda()                        

    Token_index = torch.LongTensor([1]).cuda()
   
    
########################################################################################################
    def speak(text):
        
        import subprocess

         
        model = "en_US-kathleen-low.onnx"    # Update path to Piper model, u can choose other language i.e. Arabic, French, ...

         
        output_file = "text.wav"

        # Command to be executed
        command = f'echo "{text}" | piper.exe -m {model} -f {output_file}'

        # Run the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Print the output of the command
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

        from playsound import playsound

        playsound('text.wav')

######################################################################################################## 
    
    while(True):
        
        c += 1
           
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
 
        last_6_frame.append(frame)
        
        if len(last_6_frame) == 6:  # this mean sampling at 5 fps (30/5=6)
            
                 last_6_frame = []
                 frame = frame[:, :, ::-1]  # Convert BGR to RGB
                
                
                 frame = frame / 255.0
                 frame1 = (frame - mean) / std
                 frame1 = np.transpose(frame1, (2, 0, 1))
                 frame_list1.append(frame1)
                 
                 
                 frame_list2.append(frame)
               
       
                  
                 if len(frame_list1) == 8: 

  
                     count += 1
                     
                     
                     frame_list1 = np.array(frame_list1, dtype=np.float32)
               
                      
                     frame_list1 = torch.stack([torch.tensor(frame2).float() for frame2 in frame_list1]).cuda()
                     
                     
                     frame_list2 = (frame_list2 - np.array(mean)) / np.array(std)
                      
                     frame_list2 = torch.stack([torch.tensor(frame2).float() for frame2 in frame_list2])
                    
                          
                     frame_list2 = frame_list2.permute(3, 0, 1, 2).unsqueeze(0).cuda()
                     
                     
                     with torch.no_grad():

                         start_time = time.time()      #################     time

                         B3 = model1(frame_list1).unsqueeze(0) 
                
 
                      
                         feats_swin = model2.extract_feat(frame_list2).mean(axis=(2, 3, 4))[:, np.newaxis, ...]
                         
                 
                         feats_swin = feats_swin.repeat(1, 8, 1)
                  
                         
                         feats = torch.cat((B3, feats_swin), dim=2)
                  
 
                         captions = model.describe(scores, decoder_input, beam_indices, token_indices,Token_index,feats, beam_width=BEAM, beam_alpha=beam_alpha)
                       
                         captions = [idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions]
                         captions = ' '.join(captions) # convert list object to string



                   
                         end_time = time.time()
                         elapsed_time = end_time - start_time
                         elapsed_time = elapsed_time * 1000 
                       #  print(f"Time cap ALL========= : {elapsed_time:.2f} ")
                         print(  count,'=' ,captions)
                         speak(captions)
                   
                         frame_list1 = []
                         frame_list2 = []
                        
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
  
    return captions


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence
 

def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.decoder.load_state_dict(checkpoint['decoder'])
    
    return model

  

############################################################################################################## 
#############################################################################################################
################################################################################################################
################################################################################################################## 
 
 
import torch.nn as nn
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs.train_setting", help="module path of the configuration file.")
    return parser.parse_args()


 
import pickle
 

def load_vocab(filename):
    with open(filename, 'rb') as file:
        vocab = pickle.load(file)
    return vocab

  
def build_model(C, vocab):
    decoder = Transformer(                  # Decoder come from: from models.decoder import Decoder
       
        output_size=vocab.n_vocabs,                   
        d_feat_video = C.decoder.d_feat_video,
        d_model_video = C.decoder.embed_size, 
      
        d_ff_video = C.decoder.ff_dim, 
   
        
        H = C.decoder.num_heads ,  
        N_video = C.decoder.N,                  
        dout_p = C.decoder.dropout, 
        use_linear_embedder = C.decoder.use_linear_embedder, 
        use_encoder = C.decoder.use_encoder, 
        
        )                
        
         
    model = CaptionGenerator(decoder, C.loader.max_caption_len, vocab, vocab.n_vocabs)
    model.eval()
    model.cuda()
    
    
    for p in model.parameters():
        if p.dim() > 1:
           nn.init.xavier_uniform_(p)
        
    return model




  

def speak(text):
    
    import subprocess

     
    model = "en_US-kathleen-low.onnx"   

     
    output_file = "text.wav"

    # Command to be executed
    command = f'echo "{text}" | piper.exe -m {model} -f {output_file}'

    # Run the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output of the command
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    from playsound import playsound

    playsound('text.wav')

import os
import queue
import sounddevice as sd
import vosk
import json
import sys 
 
def main():
    
     
    
    # Set the model path for VOSK recognizer
    model_path = "vosk-model-small-en-us-0.15"  # Update this to your model path

    # Initialize the recognizer
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        exit(1)

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, 16000)

    # Queue to hold audio data
    q = queue.Queue()

    # Audio callback function
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    # Start the stream
    def recognize_single_word():
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            text = 'Listening for your command'
            speak(text)

            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    result_dict = json.loads(result)
                    print(result_dict)
                    if 'text' in result_dict:
                        text = result_dict['text']
                        if "detect" in text:                           
                            detect()
                                
                        elif "describe" in text:
                            args = parse_args()
                            C = importlib.import_module(args.config).TrainConfig # = import configs.train_stage1.TrainConfig
                             
                           
                            vocab = load_vocab('vocab_min_5_max_15.pkl')
                             
                            model = build_model(C, vocab)
                         
                            best_ckpt_fpath = './checkpoints/epoch_21_effeB3-Swin-S__1024_dim.ckpt'  # checpoint not for same features
                            
                            best_model = load_checkpoint(model, best_ckpt_fpath)
                            
                              
                            video_description(best_model, best_model.vocab)
                        
                        else:
                            text = 'Unknown command, please repeat your command'
                            speak(text)
                else:
                    text = 'Unknown command, please repeat your command'
                    speak(text)
         
    # Main loop
    while True:
        recognize_single_word()
    

     
if __name__ == "__main__":
    main()

