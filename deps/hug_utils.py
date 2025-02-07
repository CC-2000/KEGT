from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, OPTForCausalLM
from typing import Dict, List, Tuple

import torch
import unicodedata
import os

def top_k_sampling(softmax_out, k=5):
    """
    also named nucleus sampler
    """
    
    # [b, top_k]
    tk = torch.topk(softmax_out, k, dim=1).indices  # 可以通过.values得到top_k值

    # [b, top_k]
    softmax_out_top_k = torch.gather(softmax_out, 1, tk)    # 等价于torch.topk(softmax_out, top_k, dim=1).values
    # [b, top_k]
    softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

    # [b, 1]
    new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
    # [b, 1]
    new_toks = torch.gather(tk, 1, new_tok_indices)
    
    return new_toks

def greedy_sampling(softmax_out):
    return top_k_sampling(softmax_out, 1)

def top_p_sampling(softmax_out, p=0.9):

    softmax_out = torch.nn.functional.softmax(softmax_out, dim=1)
    sorted_probs, indices = torch.sort(softmax_out, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_sum_probs < p
    top_p_probs = sorted_probs.masked_fill(~mask, 0)
    new_tok_indices = torch.multinomial(top_p_probs, 1)
    new_toks = torch.gather(indices, 1, new_tok_indices)
    
    return new_toks


DECODING_ALG = [
    'top_k_sampling',
    'greedy_sampling',
    'top_p_sampling'
]

DECODING_ALG_FOR_STATUS_PATH = [
    'greedy_sampling',
    'greedy',
]

def decoding(decoding_alg, softmax_out, **kwargs):
    if decoding_alg == 'top_k_sampling':
        return top_k_sampling(softmax_out, k=kwargs['k'])
    elif decoding_alg == 'greedy_sampling':
        return greedy_sampling(softmax_out)
    elif decoding_alg == 'top_p_sampling':
        return top_p_sampling(softmax_out, p=kwargs['p'])


 
def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    max_len: int = 100,
    decoding_alg: str = 'top_k',
    only_response: bool=False,
    **kwargs
):
    
    assert decoding_alg in DECODING_ALG, f'decoding algorithm only supports {DECODING_ALG.keys()}'
    if decoding_alg == 'top_k_sampling':
        assert 'k' in kwargs.keys(), 'top_k sampling needs parameter k'
    elif decoding_alg == 'top_p_sampling':
        assert 'p' in kwargs.keys(), 'top_p sampling needs parameter p'
   
    
    inps = tok(prompts, padding=True, return_tensors='pt').to(model.device)

    input_ids, attention_mask = inps['input_ids'], inps['attention_mask']
    only_response_ids = [[] * input_ids.size(0)]
    batch_size = input_ids.size(0)
    past_key_values = None

    cur_context = slice(0, attention_mask.sum(1).min().item())
    
    raw_attention_mask = attention_mask.clone()
    attention_mask = attention_mask[:, cur_context].clone()
    with torch.no_grad():
        while input_ids.size(1) < max_len:  
            outs = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outs.logits
            past_key_values = outs.past_key_values
            
            
            # [b, vob_size]
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            
            new_toks = decoding(decoding_alg, softmax_out, **kwargs)
            # # [b, top_k]
            # tk = torch.topk(softmax_out, top_k, dim=1).indices  # 可以通过.values得到top_k值

            # # [b, top_k]
            # softmax_out_top_k = torch.gather(softmax_out, 1, tk)    # 等价于torch.topk(softmax_out, top_k, dim=1).values
            # # [b, top_k]
            # softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            # # [b, 1]
            # new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            # # [b, 1]
            # new_toks = torch.gather(tk, 1, new_tok_indices)
            
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [input_ids, input_ids.new_ones(batch_size, 1) * tok.pad_token_id], dim=1,
                )
            else:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
            
            last_no_masked_index = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_no_masked_index[i] + 1
                if new_idx.item() >= raw_attention_mask[i].sum() and new_idx < max_len:
                    input_ids[i][new_idx] = new_toks[i]
                    only_response_ids[i].append(new_toks[i])
                    attention_mask[i][new_idx] = 1
                elif new_idx < max_len:
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace('</s>', "")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt


def generate_fast_for_next_k_token(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    k: int,
    decoding_alg: str = 'greedy_sampling',
    **kwargs
):
    
    assert decoding_alg in DECODING_ALG, f'[decoding algorithm for status path only supports] {DECODING_ALG.keys()}'
   
    
    inps = tok(prompts, padding=True, return_tensors='pt').to(model.device)

    input_ids, attention_mask = inps['input_ids'], inps['attention_mask']
    batch_size = input_ids.size(0)

    past_key_values = None
    cur_context = slice(0, attention_mask.sum(1).min().item())
    
    raw_attention_mask = attention_mask.clone()
    max_len = raw_attention_mask.sum(dim=1).max() + k
    attention_mask = attention_mask[:, cur_context].clone()
    with torch.no_grad():
        while input_ids.size(1) < max_len:  
            outs = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outs.logits
            past_key_values = outs.past_key_values
            
            
            # [b, vob_size]
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            
            new_toks = decoding(decoding_alg, softmax_out, **kwargs)
            # # [b, top_k]
            # tk = torch.topk(softmax_out, top_k, dim=1).indices  # 可以通过.values得到top_k值

            # # [b, top_k]
            # softmax_out_top_k = torch.gather(softmax_out, 1, tk)    # 等价于torch.topk(softmax_out, top_k, dim=1).values
            # # [b, top_k]
            # softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            # # [b, 1]
            # new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            # # [b, 1]
            # new_toks = torch.gather(tk, 1, new_tok_indices)
            
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [input_ids, input_ids.new_ones(batch_size, 1) * tok.pad_token_id], dim=1,
                )
            else:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
            
            last_no_masked_index = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_no_masked_index[i] + 1
                if new_idx.item() >= raw_attention_mask[i].sum() and new_idx < max_len:
                    input_ids[i][new_idx] = new_toks[i]
                    # only_response_ids[i].append(new_toks[i])
                    attention_mask[i][new_idx] = 1
                elif new_idx < max_len:
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace('</s>', "")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt

def generate_next_k(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    k: int,
    decoding_alg: str = 'greedy_sampling',
    **kwargs
):
    
    assert decoding_alg in DECODING_ALG, f'[decoding algorithm for status path only supports] {DECODING_ALG.keys()}'
   
    
    inps = tok(prompts, padding=True, return_tensors='pt').to(model.device)

    input_ids, attention_mask = inps['input_ids'], inps['attention_mask']
    batch_size = input_ids.size(0)
    original_prompt_length = input_ids.size(1)

    past_key_values = None
    cur_context = slice(0, attention_mask.sum(1).min().item())
    
    raw_attention_mask = attention_mask.clone()
    max_len = raw_attention_mask.sum(dim=1).max() + k
    attention_mask = attention_mask[:, cur_context].clone()
    with torch.no_grad():
        while input_ids.size(1) < max_len:  
            outs = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outs.logits
            past_key_values = outs.past_key_values
            
            
            # [b, vob_size]
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            
            new_toks = decoding(decoding_alg, softmax_out, **kwargs)
            # # [b, top_k]
            # tk = torch.topk(softmax_out, top_k, dim=1).indices  # 可以通过.values得到top_k值

            # # [b, top_k]
            # softmax_out_top_k = torch.gather(softmax_out, 1, tk)    # 等价于torch.topk(softmax_out, top_k, dim=1).values
            # # [b, top_k]
            # softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            # # [b, 1]
            # new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            # # [b, 1]
            # new_toks = torch.gather(tk, 1, new_tok_indices)
            
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [input_ids, input_ids.new_ones(batch_size, 1) * tok.pad_token_id], dim=1,
                )
            else:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
            
            last_no_masked_index = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_no_masked_index[i] + 1
                if new_idx.item() >= raw_attention_mask[i].sum() and new_idx < max_len:
                    input_ids[i][new_idx] = new_toks[i]
                    # only_response_ids[i].append(new_toks[i])
                    attention_mask[i][new_idx] = 1
                elif new_idx < max_len:
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    original_txt = []
    generate_txt = []
    for i in range(len(input_ids[0])):
        if i < original_prompt_length:
            original_txt.append(tok.decode(input_ids[0][i]))
        else:
            generate_txt.append(tok.decode(input_ids[0][i]))
    # original_txt = [''.join(original_txt)]
    # generate_txt = [''.join(generate_txt)]
    original_txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace('</s>', "")
        .replace("<|endoftext|>", "")
        for x in original_txt
    ]
    generate_txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace('</s>', "")
        .replace("<|endoftext|>", "")
        for x in generate_txt
    ]
    # print(original_txt)
    # print(generate_txt)
    # txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    # txt = [
    #     unicodedata.normalize("NFKD", x)
    #     .replace("\n\n", " ")
    #     .replace("\n", " ")
    #     .replace('</s>', "")
    #     .replace("<|endoftext|>", "")
    #     for x in txt
    # ]

    return original_txt, generate_txt

def get_model_layers(model_name, model):

    ret = None
    if 'opt' in model_name:
        ret = len(model.model.decoder.layers)
    elif 'gpt2' in model_name:
        ret = len(model.transformer.h)
    elif 'Qwen' in model_name:
        ret = len(model.model.layers)
    elif 'llama-2' in model_name:
        ret = len(model.model.layers)
    elif 'llama3.1' in model_name:
        ret = len(model.model.layers)

    return ret


def load_model(model_hub_root, model_name, device):
    model_path = os.path.join(model_hub_root, model_name)

    model_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': device
    }
    
    if 'llama' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map=device) # device
    elif 'gpt2-xl' in model_name or 'Qwen2.5-1.5B' in model_name or 'opt-1.3b' in model_name:
        # 半精度
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    elif '7B' in model_name or '7b' in model_name or 'opt-6.7b' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map=device) # device
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    if 'opt-13b' in model_name:
        tok = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True, use_fast=False)
    else:
        tok = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
    
    if 'gpt2' or 'llama3.1' in model_name:
        tok.pad_token = tok.eos_token
    
    return model, tok
        