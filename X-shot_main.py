import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manager import Wiki_Data_Manager, DataManager
import nltk
import argparse
import pandas as pd
import random
from sklearn.manifold  import TSNE
import logging
from pprint import pprint

from utils import plot, get_pcs, load_acts, ConfusionMatrix
from model import LRProbe
import deps

from main import get_known_id_list


def make_evaluate_prompt(config, prompt, token, label):
    query = config['prompt_template'].format(prompt, token)
    if label ==1 :
        str_ans = "Yes, it will appear in my answer."
    else:
        str_ans = "No, it will not appear in my answer."
    evaluate_query = "Q: {}\nA:{}\n\n".format(query, str_ans)
    return evaluate_query

def main_few(config):
    model_hub_path = config['model_hub_path']
    cache_path = config['cache_path']
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']

    device = deps.get_device(config['device_id'])
    model, tok = deps.load_model(model_hub_path, model_name, device)

    # split
    known_id_list = get_known_id_list(os.path.join(data_root, dataset_name + '.json'))
    known_id_list = random.sample(known_id_list, len(known_id_list))
    trainset_number = int(len(known_id_list) * (1 - config['test_size']))
    train_dataset = known_id_list[:trainset_number]
    test_dataset  = known_id_list[trainset_number:]
    if config['save_logging']:
        logging.info("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))
    print("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))

    # data manager init
    data_manager = DataManager(cache_path, dataset_name, './data/chen/', model_name)
    confusionMatrix = ConfusionMatrix()
    prompt_template = config['prompt_template']

    for known_id in tqdm(test_dataset, total=len(test_dataset)):
        known_id = int(known_id)
        df = data_manager.df.groupby('known_id').get_group(known_id).copy()

        for i in range(len(df)):
            cur_data = df.iloc[i]
            label = cur_data['label']
            
            query = prompt_template.format(cur_data['prompt'], cur_data['token'])
            
            context_idx_list = random.sample(range(len(data_manager.df)), k=config['k_shot'])
            evaluate_query = ''
            for context_idx in context_idx_list:
                cur_query = make_evaluate_prompt(
                    config,
                    prompt=data_manager.df.iloc[context_idx]['prompt'],
                    token=data_manager.df.iloc[context_idx]['token'],
                    label=data_manager.df.iloc[context_idx]['label']
                )
                evaluate_query = evaluate_query + cur_query
            evaluate_query = evaluate_query + query

            ans = deps.generate_next_k(model, tok, [evaluate_query], config['next_k'])[1]
            ans = list(map(lambda x: x.lower().strip(), ans))
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            if label == 1:
                if "yes" in ans:
                    tp += 1
                else:
                    fn += 1
            else:
                if 'no' in ans:
                    tn += 1
                else:
                    fp += 1

            confusionMatrix.add(tp, fp, tn, fn)

    if config['save_logging']:
        logging.info("Results few-shot")
        logging.info(confusionMatrix.__str__())
    deps.print_loud(confusionMatrix.__str__())


def main_zero(config):
    model_hub_path = config['model_hub_path']
    cache_path = config['cache_path']
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    device = deps.get_device(config['device_id'])
    model, tok = deps.load_model(model_hub_path, model_name, device)

    # split
    known_id_list = get_known_id_list(os.path.join(data_root, dataset_name + '.json'))
    known_id_list = random.sample(known_id_list, len(known_id_list))
    trainset_number = int(len(known_id_list) * (1 - config['test_size']))
    train_dataset = known_id_list[:trainset_number]
    test_dataset  = known_id_list[trainset_number:]
    if config['save_logging']:
        logging.info("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))
    print("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))


    # data manager init
    data_manager = DataManager(cache_path, dataset_name, './data/chen/', model_name)
    confusionMatrix = ConfusionMatrix()
    prompt_template = config['prompt_template']
    for known_id in tqdm(test_dataset, total=len(test_dataset)):
        known_id = int(known_id)
        df = data_manager.df.groupby('known_id').get_group(known_id).copy()
        
        for i in range(len(df)):
            cur_data = df.iloc[i]
            label = cur_data['label']
            
            query = prompt_template.format(cur_data['prompt'], cur_data['token'])
            
            ans = deps.generate_next_k(model, tok, [query], config['next_k'])[1]
            ans = list(map(lambda x: x.lower().strip(), ans))
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            if label == 1:
                if "yes" in ans:
                    tp += 1
                else:
                    fn += 1
            else:
                if 'no' in ans:
                    tn += 1
                else:
                    fp += 1

            confusionMatrix.add(tp, fp, tn, fn)
    
    if config['save_logging']:
        logging.info("Results zero-shot")
        logging.info(confusionMatrix.__str__())
    deps.print_loud(confusionMatrix.__str__())


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='few-shot_exp.yaml')
    parser.add_argument("--model_name", type=str, default='gpt2-medium')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_arg_parser()
    config_path = os.path.join('config', args.model_name, args.config_file)
    config = deps.load_yaml(config_path)

    assert args.model_name == config['model_name']

    deps.set_seed(config['seed'])
    current_time = deps.get_cur_time()
    config['current_time'] = current_time

    is_zero = 'zero-shot' in args.config_file

    if config['save_logging']:

        if is_zero:
            logging_path = os.path.join(config['cache_path'], 'logs', config['model_name'], 'zero-shot', current_time + '/')
        else:
            logging_path = os.path.join(config['cache_path'], 'logs', config['model_name'], '{}-shot'.format(config['k_shot']), current_time + '/')

        deps.create_folders(logging_path)
        config['logging_path'] = logging_path
        logging.basicConfig(filename=os.path.join(logging_path, 'logging.log'), filemode='w', format='%(asctime)s %(name)s:%(levelname)s:%(message)s', datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        logging.info(config)

        deps.save_yaml(os.path.join(logging_path, 'zero-shot_exp.yaml'), config)        
        deps.print_loud("The log is saved in path {}".format(logging_path))

    if is_zero:
        main_zero(config)
    else:
        main_few(config)