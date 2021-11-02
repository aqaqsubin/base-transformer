import argparse
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from dataset import MathData
from transformer.encoder_decoder import Encoder, Decoder
from transformer.util import create_padding_mask, create_look_ahead_mask


class LightningTransformer(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(LightningTransformer, self).__init__()
        self.hparams = hparams
        self.neg = -0.7
        self.transformer = Transformer(
                            num_layers=hparams.num_layers,
                            d_model=hparams.d_model,
                            num_heads=hparams.num_heads,
                            dff=hparams.dff,
                            input_vocab_size=hparams.vocab_size,
                            target_vocab_size=hparams.vocab_size,
                            pe_input=hparams.max_len,
                            pe_target=hparams.max_len-1,
                            device=kwargs["device"],
                            rate=hparams.dropout_rate
                        )
        self.tok = kwargs['tokenizer']
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_len',
                            type=int,
                            default=120)

        parser.add_argument('--batch_size',
                            type=int,
                            default=32)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        parser.add_argument('--num_heads',
                            type=int,
                            default=8)
        parser.add_argument('--num_layers',
                            type=int,
                            default=6)
        parser.add_argument('--d_model',
                            type=int,
                            default=512)
        parser.add_argument('--dff',
                            type=int,
                            default=2048)
        parser.add_argument('--dropout_rate',
                            type=float,
                            default=0.1)
        parser.add_argument('--vocab_size',
                            type=float,
                            default=51200)
        return parser

    def forward(self, enc_input, dec_input, enc_output=None, output_attentions=False):
        logits, attn, enc_out = self.transformer([enc_input, dec_input, enc_output])
        
        if output_attentions:
            return (logits, attn, enc_out)
        
        return logits, enc_out

    def accuracy_function(self, real, pred):
        accuracies = torch.eq(real, torch.argmax(pred, dim=2))
        mask = torch.logical_not(torch.eq(real, self.tok.pad_token_id))
        accuracies = torch.logical_and(mask, accuracies)
        accuracies = torch.tensor(accuracies, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return torch.sum(accuracies)/torch.sum(mask)

    def cross_entropy_loss(self, real, pred):
        mask = torch.tensor(torch.logical_not(torch.eq(real, self.tok.pad_token_id)), dtype=pred.dtype)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=pred.shape[2], dim=2)
        
        negative_logits = self.neg * torch.ones_like(pred) * torch.finfo(pred.dtype).max
        mask_out = torch.where(mask_3d == 1, pred, negative_logits)
        loss = self.criterion(mask_out.transpose(2, 1), real)
        loss_avg = loss.sum() / mask.sum()
        return loss_avg

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        out, _ = self(enc_input=src, dec_input=tgt[:, :-1])
        train_loss = self.cross_entropy_loss(real=tgt[:, 1:], pred=out)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        out, _ = self(enc_input=src, dec_input=tgt[:, :-1])
        val_loss = self.cross_entropy_loss(real=tgt[:, 1:], pred=out)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        src = [item[0] for item in batch]
        tgt = [item[1] for item in batch]
        return torch.LongTensor(src), torch.LongTensor(tgt)

    def train_dataloader(self):
        self.train_set = MathData(filepath=f'{self.hparams.data_dir}/train.csv', target=self.hparams.target, max_len=self.hparams.max_len, tokenizer=self.tok)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader
    
    def val_dataloader(self):
        self.valid_set = MathData(filepath=f'{self.hparams.data_dir}/val.csv', target=self.hparams.target, max_len=self.hparams.max_len, tokenizer=self.tok)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return val_dataloader


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, device, rate=0.1):
        super().__init__()
        self.device = device
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, device, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, device, rate)

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inputs):
        inp, tar, enc_output = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, enc_padding_mask, enc_output)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights, enc_output

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tar.size(1))
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = torch.maximum(dec_target_padding_mask.to(self.device), look_ahead_mask.to(self.device))

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

