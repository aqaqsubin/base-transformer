
# -*- coding: utf-8 -*-
import re
import torch
import pandas as pd
from torch import nn

from os.path import join as pjoin
from lightning_transformer import LightningTransformer
from data_utils import read_df, saveCSVFile

def base_setting(args):
    args.max_len = getattr(args, 'max_len', 120)
    args.batch_size = getattr(args, 'batch_size', 4)
    args.log = getattr(args, 'log', True)
    args.num_heads = getattr(args, 'num_heads', 8)
    args.num_layers = getattr(args, 'num_layers', 6)
    args.d_model = getattr(args, 'd_model', 512)
    args.dff = getattr(args, 'dff', 2048)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.vocab_size = getattr(args, 'vocab_size', 51200)

def tokenize_txt_input(tokenizer, input_txt, max_len):
    q_toked = tokenizer.tokenize(tokenizer.bos_token + input_txt + tokenizer.eos_token)
    
    if len(q_toked) > max_len:
        q_toked = q_toked[:max_len-1] + q_toked[-1]

    token_ids = tokenizer.convert_tokens_to_ids(q_toked)
    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]

    return token_ids


def evaluation(args, **kwargs):
    # load params
    base_setting(args)
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    print(args.model_pt)
    print(args.num_layers)
    # model = KoGPT2Chat(args)
    # model = model.load_from_checkpoint(args.model_pt)

    model = LightningTransformer(args, device=torch.device(device), tokenizer=kwargs['tokenizer'])
    if args.cuda:
        model = model.cuda()

    if args.model_pt is not None:
        if args.model_pt.endswith('ckpt'):
            model = LightningTransformer.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, device=torch.device(device), tokenizer=kwargs['tokenizer'])
        elif args.model_pt.endswith('bin'):
            state_dict = torch.load(args.model_pt, map_location=device)
            own_state = model.state_dict()
            print(own_state.items())
            for name, param in state_dict.items():
                print("name : {}".format(name))
                if name not in own_state:
                    print("drop : {}".format(name))
                    continue
                if isinstance(param, nn.Parameter):
                    param = param.data
                print("copying name : {}".format(name))
                own_state[name].copy_(param)
        else:
            raise TypeError('Unknown file extension')

    if args.cuda:
        model = model.cuda()     

    model.eval()

    test_df = read_df(pjoin(args.data_dir, 'test.csv'))
    eval_dist = pd.DataFrame(columns=['question', 'gen_equation', 'tgt_equation', 'is_right'])

    arithmetic_op = ('round', 'floor', 'ceil', 'abs', 'add', 'sub', 'div', 'mul', 'quo', 'mod', 'pow')
    is_arithmetic = lambda x: x.startswith(arithmetic_op) or x[0].isdecimal() or re.match('[A-Z]', x[0]) is not None or x.startswith('(')
    


    with torch.no_grad():
        for row in test_df.iterrows():

            question = row[1].question
            target = row[1][args.target]
            answer = ''
            if not re.sub('\s+', '', question):
                print("Drop %d row (Empty row)" % row[0])
                continue
            
            print("Question: %s" % question)

            enc_input = torch.LongTensor(tokenize_txt_input(model.tok, input_txt=question, max_len=args.max_len)).unsqueeze(0).to(device=device)
            enc_output = None

            answer = ''
            
            for iter_ in range(args.max_len-1):
                answer_tok = model.tok.tokenize(model.tok.bos_token + answer)
                output = torch.LongTensor(model.tok.convert_tokens_to_ids(answer_tok) * enc_input.size(0)).unsqueeze(0).to(device)
                
                #torch.LongTensor(model.tok.convert_tokens_to_ids(answer_tok)).to(device=device)
                
                logits, enc_output = model(enc_input=enc_input, dec_input=output, enc_output=enc_output, output_attentions=False)
                logits = logits[: ,-1:, :]

                gen = model.tok.convert_ids_to_tokens(torch.argmax(logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == model.tok.eos_token:
                    break
                answer += re.sub('[#]+', '', gen)

            answer= answer.strip()
            target = str(target).strip()
            
            print("Answer: {}".format(answer))
            input_len = enc_input.size(1)
            
            is_arithmetic_op = is_arithmetic(answer)

            proc_target = re.sub('\s+', '', target)
            proc_answer = re.sub('\s+', '', answer)

            dic = {'question' : question, 'gen_equation' : answer, 'tgt_equation' : target, 
            'is_right' : proc_target == proc_answer, 'input_len' : input_len, 'is_arithmetic_op' : is_arithmetic_op}
            eval_dist = eval_dist.append(dic, ignore_index=True)

        saveCSVFile(pjoin(args.save_dir, f'eval_{args.model_name}.csv'), eval_dist)
            

