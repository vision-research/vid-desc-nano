import inspect
import os
import argparse 
import importlib
import torch
import torch.nn as nn
from tqdm import tqdm
 

import losses
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from models.decoder import mask

alpha = 0.5
BEAM = 3
LBS = 0.7

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs.train_setting", help="module path of the configuration file.")
    return parser.parse_args()

args = parse_args()
C = importlib.import_module(args.config).TrainConfig 



def org_captions(captions_dict):
   

    # Loop through each key-value pair in the dictionary
    for key, caption_list in captions_dict.items():
        # Process each caption in the list
        processed_captions = []
        for caption in caption_list:
            # Preprocess the caption
           
            processed_captions.append(caption)
        # Replace the original caption list with the processed one
        captions_dict[key] = processed_captions

    return captions_dict

def pred_captions(captions_dict):
     

    # Process each caption in the dictionary
    for key, caption in captions_dict.items():
        # Preprocess the caption
        
        # Replace the original caption with the processed one
        captions_dict[key] = caption

    return captions_dict

 
def create_caption_dictionary_from_txt_file(file_path):
    # Initialize an empty dictionary to store the captions
    caption_dict = {}

    # Open the text file for reading
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in file:
            # Split each line into video ID and caption
            video_id, caption_arabic = line.strip().split(' ', 1)
            
          #  caption_arabic = remove_harakat(caption_arabic)

            # Check if the video ID is already in the dictionary
            if video_id in caption_dict:
                caption_dict[video_id].append(caption_arabic)
            else:
                # If not, create a new entry with a list containing the caption
                caption_dict[video_id] = [caption_arabic]

    return caption_dict

 

def parse_batch(batch):
    vids, feats, captions = batch
    feats = [feat.cuda() for feat in feats]
    feats = torch.cat(feats, dim=2)
    captions = captions.long().cuda()
    return vids, feats, captions


def set_up_causal_mask(seq_len):  # , device):
    """Defines the triangular mask used in transformers.
    This mask prevents decoder from attending the tokens after the current one.
    Arguments:
        seq_len (int): Maximum length of input sequence
        device: Device on which to map the created tensor mask
    Returns:
        mask (torch.Tensor): Created triangular mask
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.bool().masked_fill(mask == 0, bool(
        '-inf')).masked_fill(mask == 1, float(0.0))   
    mask.requires_grad = False
    return mask



max_caption_len = C.loader.max_caption_len
target_subsequent_mask = set_up_causal_mask(max_caption_len + 1)
target_subsequent_mask = target_subsequent_mask.cuda()
loss_fcn = nn.CrossEntropyLoss()


def train(e, model, optimizer, train_iter, vocab, reg_lambda, recon_lambda, gradient_clip):
    model.train()
    train_total_loss = 0
  

    
    PAD_idx = vocab.word2idx['<PAD>']
    t = tqdm(train_iter)
    for batch in t:

        _, feats, captions = parse_batch(batch)
        captions2 = captions.permute(1, 0)
        captions3 = captions.permute(1, 0)
        captions2 = captions2[:, :-1]
        captions3 = captions3[:, 1:]
  
  
        optimizer.zero_grad()
 
  
        masks = mask(feats[:, :, 0], captions2, PAD_idx)
     
        output = model(feats, captions2, masks)
        
 
      
        
        criterion = losses.LabelSmoothing(LBS, PAD_idx)
        n_tokens = (captions3 != PAD_idx).sum() # calculate the number of False , which represent the real caption not padded
        
        loss = criterion(output, captions3) / n_tokens
        
      
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
        optimizer.step()
        
        train_total_loss += loss.item()

        
        t.set_description("[Epoch #{0}] loss: {1:.3f}".format(e,  loss))

    train_total_loss_norm = train_total_loss / len(train_iter)
    
     
     
    loss = {
        'total': train_total_loss_norm
       
    }
    return loss


def test(model, val_iter, vocab, reg_lambda, recon_lambda):
    model.eval()
    val_total_loss = 0

   
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(val_iter, 1):
        _, feats, captions = parse_batch(batch)
        captions2 = captions.permute(1, 0)
        captions3 = captions.permute(1, 0)
        captions2 = captions2[:, :-1]
        captions3 = captions3[:, 1:]
        captions3 = captions3.permute(1, 0)

        target_pad_mask = (captions2 == PAD_idx)
        target_pad_mask = target_pad_mask.cuda()
     
      
        with torch.no_grad():      
            
            masks = mask(feats[:, :, 0], captions2, PAD_idx)
            output = model(feats, captions2, masks)
          
         

            criterion = losses.LabelSmoothing(LBS, PAD_idx)
            
            n_tokens = (captions3 != PAD_idx).sum()
            
            loss = criterion(output, captions3) / n_tokens

          
            val_total_loss += loss.item()
      
        
    loss = {
        'total': loss
            }
    
    return loss


def get_predicted_captions(data_iter, model, vocab, beam_width=BEAM, beam_alpha=alpha): # data_iter here is validation data
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _ = parse_batch(batch)
            for vid, feat in zip(vids, feats):
                if vid not in onlyonce_dataset:
                    onlyonce_dataset[vid] = feat
        onlyonce_iter = []

        
        vids = list(onlyonce_dataset.keys())
        feats = list(onlyonce_dataset.values())
 
        batch_size = 1  # very affect on result
     
        while len(vids) > 0:
            
            onlyonce_iter.append(( vids[:batch_size], torch.stack(feats[:batch_size]) ))

            
            vids = vids[batch_size:]
            feats = feats[batch_size:]
        
     
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter)

    vid2pred = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for vids, feats in onlyonce_iter:
    
        captions = model.describe(feats, beam_width=BEAM, beam_alpha=beam_alpha)
    
        captions = [ idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions ]
      
        vid2pred.update({ v: p for v, p in zip(vids, captions) })
  
     
    return vid2pred


def get_groundtruth_captions(data_iter, vocab):
    vid2GTs = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for batch in iter(data_iter):
        vids, _, captions = parse_batch(batch)
        captions = captions.transpose(0, 1)
        for vid, caption in zip(vids, captions):
            if vid not in vid2GTs:
                vid2GTs[vid] = []
            caption = idxs_to_sentence(caption, vocab.idx2word, EOS_idx)
            vid2GTs[vid].append(caption)
    return vid2GTs


def score(vid2pred, vid2GTs): # for disply video name instead vidoe number 1 2 ...
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    len_test = len(vid2pred) 
    vid2GTs = org_captions(vid2GTs)
    vid2pred = pred_captions(vid2pred)
  #  refs = {vid: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid: [pred] for vid, pred in vid2pred.items()}
    
      
    
    if len_test < 110:  # val set
        
        file_path = 'data/MSVD/metadata/validation.txt'
        refs = create_caption_dictionary_from_txt_file(file_path)
        print(' ==================== validation phase ================================='   ) 
        
  
    if len_test > 600:   # test set
            
        file_path = 'data/MSVD/metadata/test.txt'
        refs = create_caption_dictionary_from_txt_file(file_path)
        print(' ==================== testing phase ================================='   ) 
     
    scores = calc_scores(refs, hypos)
    return scores

# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),   
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def evaluate(data_iter, model, vocab, beam_width=BEAM, beam_alpha=0.):
    vid2pred = get_predicted_captions(data_iter, model, vocab, beam_width=BEAM, beam_alpha=0.)
    vid2GTs = get_groundtruth_captions(data_iter, vocab)
    scores = score(vid2pred, vid2GTs)
    return scores

# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.decoder.load_state_dict(checkpoint['decoder'])
    
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'decoder': model.decoder.state_dict(), 
        'reconstructor':  None,
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([str(vid), pred, GTs])
            fout.write("{}\n".format(line))
