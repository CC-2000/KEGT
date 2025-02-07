import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
from nnsight import LanguageModel

import deps


attn_module = {
    'llama3.1-8b': 'model.layers.{}.self_attn',
    'Qwen2.5-1.5B': 'model.layers.{}.self_attn',
    'gpt2-medium': 'transformer.h.{}.attn',
}
mlp_module = {
    'llama3.1-8b': 'model.layers.{}.mlp'
}

def load_csv(data_root, dataset_name, model_name):
    """
    Load statements from csv file, return list of strings.
    """
    path = os.path.join(data_root, f"{dataset_name}_{model_name}.csv")
    deps.print_loud(path)       
    csv = pd.read_csv(path)
    # statements = dataset['statement'].tolist()
    return csv

def get_acts(statements, model, layers):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    acts = {}
    with model.trace(statements):
        for layer in layers:
            acts[layer] = model.model.layers[layer].output[0][:,-1,:].save()

    for layer, act in acts.items():
        acts[layer] = act.value
    
    return acts

def get_save_dir(args):


    layers = args.layers
    save_dir = os.path.join(args.cache_dir, f"{args.output_dir}", args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, args.datasets)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, str(known_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(f"{save_dir}/layer_{layers[0]}.pt"): # and t.load(f"{save_dir}/layer_{known_id}_{layers[0]}.pt").shape[0] == len(statements):
        return None


    return save_dir



if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_name", default="llama-2-13b-hf")
    parser.add_argument("--model_hub_path", default='/home_sda1/wjy/model_hub')
    parser.add_argument("--data_root", type=str, default="./data/KASS")
    parser.add_argument("--layers", default=[-1], nargs='+', type=int, help="Layers to save embeddings from")
    
    # 同时表明存储的是 hs or a or m
    parser.add_argument("--output_dir", default="hidden_status", choices=['attn', 'hidden_status'])
    
    
    parser.add_argument("--datasets", default='KASS')
    parser.add_argument("--cache_dir", default='/home_sda1/wjy/cache/work2_try2')
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20001202)

    args = parser.parse_args()

    deps.set_seed(args.seed)
    device = deps.get_device(args.device_id)

    model, tok = deps.load_model(args.model_hub_path, args.model_name, device)

    t.set_grad_enabled(False)

    csv = load_csv(args.data_root, args.datasets, args.model_name)
    layers = args.layers
    if layers == [-1]:
        layers = list(range(deps.get_model_layers(args.model_name, model)))
        # layers = list(range(len(model.model.layers)))

    csvgroupby = csv.groupby('known_id')
    for dataset in tqdm(csvgroupby, total=len(csvgroupby)):
        known_id = dataset[0]
        df = dataset[1]

        statements = df['statement'].to_list()

        # save_dir = os.path.join(args.cache_dir, f"{args.output_dir}", args.model_name)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_dir = os.path.join(save_dir, args.datasets)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_dir = os.path.join(save_dir, str(known_id))
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # if os.path.exists(f"{save_dir}/layer_{layers[0]}.pt"): # and t.load(f"{save_dir}/layer_{known_id}_{layers[0]}.pt").shape[0] == len(statements):
        #     continue
        save_dir = get_save_dir(args)
        if save_dir is None:
            continue

        total_acts = {}
        for layer in layers:
            total_acts[layer] = []

        for idx in tqdm(range(0, len(statements))):
            if args.output_dir == 'hidden_status':
                acts = deps.collect_hiiden_states_no_nnsight([statements[idx]], model, tok, layers)
            elif args.output_dir == 'attn':
                acts = deps.collect_features_no_nnsight([statements[idx]], model, tok, attn_module[args.model_name], layers)
            for layer in layers:
                total_acts[layer].append(acts[layer])
        
        for layer, act in total_acts.items():
            final_acts = t.concat(act, dim=0).float().cpu()
            t.save(final_acts, f"{save_dir}/layer_{layer}.pt")