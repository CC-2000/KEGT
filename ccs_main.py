import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manager import Wiki_Data_Manager, DataManager, CCS_DataManager
import nltk
import argparse
import pandas as pd
import random
from sklearn.manifold  import TSNE
import logging
from pprint import pprint

from utils import plot, get_pcs, load_acts, ConfusionMatrix
from model import LRProbe, CCS
import deps

from main import get_known_id_list


def main(config):
    model_hub_path = config['model_hub_path']
    cache_path = config['cache_path']
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    device = deps.get_device(config['device_id'])
    layer = config['layer']
    model, tok = deps.load_model(model_hub_path, model_name, device)
    
    # split
    known_id_list = get_known_id_list(os.path.join(data_root, dataset_name + '.json'))
    known_id_list = random.sample(known_id_list, len(known_id_list))

    if config['mode'] == 'generalization':
        trainset_number = config['train_dataset_number']
    else:
        trainset_number = int(len(known_id_list) * (1 - config['test_size']))



    train_dataset = known_id_list[:trainset_number]
    test_dataset  = known_id_list[trainset_number:]
    if config['save_logging']:
        logging.info("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))
    print("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))

    data_manager = CCS_DataManager(cache_path, dataset_name, './data/chen/', model_name)


    for known_id in tqdm(train_dataset, total=len(train_dataset)):
        known_id = int(known_id)
        train_size = 1 - config['test_size']
        data_manager.add_dataset(model, tok, known_id, layer, train_size, device=device, target_dir=args.target_acts)
    
    for known_id in tqdm(test_dataset, total=len(test_dataset)):
        if known_id in train_dataset:
            continue
        known_id = int(known_id)
        data_manager.add_dataset(model, tok, known_id, layer, train_size=None, device=device, target_dir=args.target_acts)
    
    
    # training
    pos_acts, neg_acts, train_labels = data_manager.get('train')    
    ccs = CCS(neg_acts, pos_acts)
    ccs.repeated_train()

    # validation & iid
    pos_acts, neg_acts, cur_labels = data_manager.get('val')    
    preds = ccs.pred(neg_acts, pos_acts)
    cf_validation = ConfusionMatrix()
    cf_validation.update(cur_labels, preds)
    if config['save_logging']:
        logging.info("Results of validation & iid")
        logging.info(cf_validation.__str__())
    deps.print_loud("iid.\t" + cf_validation.__str__())


    # testset
    cf_test = ConfusionMatrix()
    for known_id in test_dataset:
        known_id = int(known_id)
        pos_acts, neg_acts, cur_labels = data_manager.get(known_id)
        preds = ccs.pred(neg_acts, pos_acts)
        cf_test.update(cur_labels, preds)
    if config['save_logging']:
        logging.info("Results of testset")
        logging.info(cf_test.__str__())
    deps.print_loud("no iid.\t" + cf_test.__str__())
        

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='ccs_exp.yaml')
    parser.add_argument("--model_name", type=str, default='gpt2-medium')
    parser.add_argument("--target_acts", type=str, default='hidden_status', choices=['hidden_status', 'attn'])
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument("--mode", type=str, default='main', choices=['main', 'generalization'])

    parser.add_argument("--train_dataset_number", type=int, default=1)
    
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
    config['mode'] = args.mode

    if args.layer != -1:
        config['layer'] = args.layer

    if config['save_logging']:

        if args.mode == 'main':
            ccs_dir = 'ccs'
        elif args.mode == 'generalization':
            ccs_dir = os.path.join('generalization', 'ccs', 'train_dataset_number_{}'.format(args.train_dataset_number))
            config['train_dataset_number'] = args.train_dataset_number

        logging_path = os.path.join(config['cache_path'], 'logs', config['model_name'], ccs_dir, '[{}]'.format(str(config['layer'])), current_time + '/')

        deps.create_folders(logging_path)
        config['logging_path'] = logging_path
        logging.basicConfig(filename=os.path.join(logging_path, 'logging.log'), filemode='w', format='%(asctime)s %(name)s:%(levelname)s:%(message)s', datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        logging.info(config)

        deps.save_yaml(os.path.join(logging_path, 'zero-shot_exp.yaml'), config)        
        deps.print_loud("The log is saved in path {}".format(logging_path))

    
    
    main(config)