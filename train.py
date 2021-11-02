import os
import datetime
import argparse
import logging
import torch
import warnings
from os.path import join as pjoin

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from eval import evaluation
from lightning_transformer import LightningTransformer

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.getcwd()
DATA_DIR = pjoin(ROOT_DIR, 'data')
MODEL_DIR = pjoin(ROOT_DIR, 'lib/model/model_ckpt')

# BOS = '<bos>'
# EOS = '<eos>'
# MASK = '<unused0>'
# PAD = '<pad>'
# UNK = '<unk>'

BOS = '[BOS]'
EOS = '[EOS]'
MASK = '[MASK]'
PAD = '[PAD]'
UNK = '[UNK]'

# TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
#             bos_token=BOS, eos_token=EOS, unk_token=UNK,
#             pad_token=PAD, mask_token=MASK) 

TOKENIZER = AutoTokenizer.from_pretrained('tunib/electra-ko-base',
                                          bos_token=BOS, eos_token=EOS, unk_token=UNK,
                                          pad_token=PAD, mask_token=MASK)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
    parser.add_argument('--model_params',
                        type=str,
                        default='model_chp1/model_-last.ckpt',
                        help='model binary for starting chat')

    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--target", choices=['equation', 'intermediate_val', 'nested_equation'], default='equation')
    parser.add_argument("--save_dir", type=str, default=f'{DATA_DIR}/result')
    parser.add_argument("--data_dir", type=str, default=f"{DATA_DIR}/preprocessed")

    parser.add_argument("--cuda", action="store_true", default=True)

    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    
    today = datetime.datetime.now()
    parser.add_argument("--model_name", type=str, default=f"{today.strftime('%m%d')}_gpt2")
    parser.add_argument("--model_pt", type=str, default=f'{MODEL_DIR}/model_last.ckpt')

    parser = LightningTransformer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logging.info(args)

    if len(args.gpuid) > 0:
        if args.eval:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args, tokenizer=TOKENIZER)

        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath='model_ckpt',
                filename='{epoch:02d}-{train_loss:.2f}',
                verbose=True,
                save_last=True,
                monitor='train_loss',
                mode='min',
                prefix='model_'
            )
            # python train_torch.py --train --gpus 1 --max_epochs 3
            model = LightningTransformer(args, device=torch.device("cuda:0"), tokenizer=TOKENIZER)
            model.train()
            trainer = Trainer(
                            check_val_every_n_epoch=1, 
                            checkpoint_callback=checkpoint_callback, 
                            flush_logs_every_n_steps=100, 
                            gpus=args.gpuid, 
                            gradient_clip_val=1.0, 
                            log_every_n_steps=50, 
                            logger=True, 
                            max_epochs=args.max_epochs, 
                            num_processes=1)
                
            trainer.fit(model)
            logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
