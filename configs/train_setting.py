import os
import time


class FeatureConfig:
    models = [ "MSVD_feature_fusion" ]  
    size = 0
    for model in models:
        if 'feature_fusion' in model:
            size += 1024
        else:
            raise NotImplementedError("Unknown model: {}".format(model))


class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 }
 

 
class MSVDLoaderConfig:
    train_caption_fpath = "data/MSVD/metadata/train.txt"        

    val_caption_fpath = "data/MSVD/metadata/validation.txt"                 
    test_caption_fpath = "data/MSVD/metadata/test.txt"              
    min_count = 5
    max_caption_len = 15   

     
   

    phase_video_feat_fpath_tpl_train_3d = "data/MSVD/features/MSVD_swin_S_8_train.hdf5"
    phase_video_feat_fpath_tpl_val_3d = "data/MSVD/features/MSVD_swin_S_8_valid.hdf5"
    phase_video_feat_fpath_tpl_test_3d = "data/MSVD/features/MSVD_swin_S_8_test.hdf5"
     
          
    
    phase_video_feat_fpath_tpl_train_2d = "data/MSVD/features/MSVD_B3_train_8.hdf5"
    phase_video_feat_fpath_tpl_val_2d = "data/MSVD/features/MSVD_B3_valid_8.hdf5"
    phase_video_feat_fpath_tpl_test_2d = "data/MSVD/features/MSVD_B3_test_8.hdf5"

 

    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
 
    
    frame_sample_len = 8

    num_workers = 4

 


class DecoderConfig:
    
    num_heads = 1
    N = 1
    dropout =  0.1
    embed_size =    1024
    ff_dim =  1024
    d_feat_video = 1024
    use_linear_embedder = True
    use_encoder = False
     
class TrainConfig:
    corpus = 'MSVD'; assert corpus in [ 'MSVD', 'MSR-VTT' ]      ######
    reconstructor_type = None; assert reconstructor_type is None

    feat = FeatureConfig
    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
         
    }[corpus]
    decoder = DecoderConfig
    reconstructor = None


    """ Optimization """
    epochs = {
        'MSVD':            50  ,
        'MSR-VTT': 1,
    }[corpus]
    batch_size          = 64
    shuffle = True
    optimizer = "AMSGrad" # AMSGrad    Adam
    gradient_clip = None # 8.0 # None if not used
    lr = {
        'MSVD': 2e-5,
       
    }[corpus]
    lr_decay_start_from = 10
    lr_decay_gamma = 0.5
    lr_decay_patience = 3
    weight_decay = 2e-5
    recon_lambda = 0.; assert recon_lambda == 0
    reg_lambda = 0

    """ Pretrained Model """
    pretrained_decoder_fpath = None
    pretrained_reconstructor_fpath = None; assert pretrained_reconstructor_fpath is None

    """ Evaluate """
    metrics = [ 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L' ]
    
    
    timestamp = time.strftime("%y%m%d_%H-%M", time.gmtime())    
    model_id = timestamp
    """ Log """
    log_dpath = "logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/decoder/cross_entropy"
    tx_train_reconstruction_loss = "loss/train/reconstructor"
    tx_train_entropy_loss = "loss/train/decoder/entropy"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/decoder/cross_entropy"
    tx_val_reconstruction_loss = "loss/val/reconstructor"
    tx_val_entropy_loss = "loss/val/decoder/entropy"
    tx_lr = "params/lr"

 