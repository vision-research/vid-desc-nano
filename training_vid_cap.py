import argparse  
import importlib
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, train, test
from loader.MSVD import MSVD  
 
from models.decoder import Transformer
from models.caption_generator import CaptionGenerator
import torch.nn as nn

 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs.train_setting", help="module path of the configuration file.")
    return parser.parse_args()

 

def build_loaders(C):   
    if C.corpus == "MSVD":
        corpus = MSVD(C)  
     
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    
 
      
    
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab

 
def build_model(C, vocab):
    decoder = Transformer(                 
       
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
    model.cuda()
    
    
    for p in model.parameters():
        if p.dim() > 1:
           nn.init.xavier_uniform_(p)
        
    return model



def log_train(C, summary_writer, e, loss, lr, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
  
    summary_writer.add_scalar(C.tx_lr, lr, e)
    print("train loss: {} ".format(loss['total']))
    if scores is not None:
      for metric in C.metrics:
          summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
      print("scores: {}".format(scores))


def log_val(C, summary_writer, e, loss, scores=None):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
   
    print("val loss: {}  ".format(loss['total']))
    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


def log_test(C, summary_writer, e, test_scores):
    for metric in C.metrics:
        summary_writer.add_scalar("TEST SCORE/{}".format(metric), test_scores[metric], e)
    print("scores: {}".format(test_scores))


def main():
    args = parse_args()
    C = importlib.import_module(args.config).TrainConfig # = import configs.train_stage1.TrainConfig
     
    print("MODEL ID: {}".format(C.model_id))

    summary_writer = SummaryWriter(C.log_dpath)

    train_iter, val_iter, test_iter, vocab = build_loaders(C)
    

    model = build_model(C, vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, eps = 1.0e-9, weight_decay=C.weight_decay, amsgrad=True)
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                     patience=C.lr_decay_patience, verbose=True)

    best_val_scores = { 'CIDEr': 0. }
    best_epoch = 0
    best_ckpt_fpath = None
    
    val_loss_list = []
    train_loss_list = []
    bleu_4_list = []
    meteor_list = []
    rouge_l_list = []
    cider_list = []
    
    
    
    for e in range(1, C.epochs + 1): # e = 1  2  3  .... counter for epochs
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)
        
        """ Train """
        print("\n")
        train_loss = train(e, model, optimizer, train_iter, vocab,
                           C.reg_lambda, C.recon_lambda, C.gradient_clip)
        log_train(C, summary_writer, e, train_loss, get_lr(optimizer))

        """ Validation """
        if e >=1:
            
            val_loss = test(model, test_iter, vocab, C.reg_lambda, C.recon_lambda)
            val_scores = evaluate(test_iter, model, model.vocab)
            log_val(C, summary_writer, e, val_loss, val_scores)
            
            
            val_loss_list.append(val_loss['total'].item())  # Convert tensor to float and add to list
            train_loss_list.append(train_loss['total'])
            bleu_4_list.append(val_scores['Bleu_4'])
            meteor_list.append(val_scores['METEOR'])
            rouge_l_list.append(val_scores['ROUGE_L'])
            cider_list.append(val_scores['CIDEr'])
            
            
            
            
            if e >= C.lr_decay_start_from:
                lr_scheduler.step(val_loss['total'])
                
            if e == 1 or val_scores['CIDEr'] > best_val_scores['CIDEr']:
                best_epoch = e
                best_val_scores = val_scores
                best_ckpt_fpath = ckpt_fpath
                
                
                if e >= C.save_from and e % C.save_every == 0:
                    print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                    save_checkpoint(e, model, ckpt_fpath, C)
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                
            
            """ Test with Best Model """
    print("\n\n\n[BEST]")
    best_model = load_checkpoint(model, best_ckpt_fpath)
    test_scores = evaluate(test_iter, best_model, best_model.vocab)
    log_test(C, summary_writer, best_epoch, test_scores)
     
if __name__ == "__main__":
     
    main()

