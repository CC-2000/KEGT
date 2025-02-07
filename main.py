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

def get_known_id_list(raw_dataset_file):
    data = deps.json_data_load(raw_dataset_file)
    known_id_list = []
    for json_data in data:
        known_id_list.append(json_data['fact_id'])
    return known_id_list

def get_public_value(cache_dir, dataset_name, model_name, layer, dataset):
    t_acts = []
    for known_id in dataset:
        known_id = int(known_id)
        acts = load_acts(cache_dir, dataset_name, model_name, known_id, layer)
        t_acts.append(acts)
    t_acts = torch.concat(t_acts, dim=0)

    public_mean = torch.mean(t_acts, dim=0)
    public_std = torch.std(t_acts, dim=0)
    return public_mean, public_std


def main(config):

    cache_path = config['cache_path']
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    layer = config['layer']
    device = deps.get_device(config['device_id'])
    

    # split
    known_list_path = os.path.join(data_root, dataset_name + '.json')
    known_id_list = get_known_id_list(known_list_path)
    # deps.print_loud("dataset path: {}".format(known_list_path))
    # logging.info("dataset path: {}".format(known_list_path))
    known_id_list = random.sample(known_id_list, len(known_id_list))

    if config['mode'] == 'generalization':
        trainset_number = config['train_dataset_number']
    elif config['mode'] == 'layer-fewshot':
        trainset_number = 3
    else:
        trainset_number = int(len(known_id_list) * (1 - config['test_size']))


    train_dataset = known_id_list[:trainset_number]
    test_dataset  = known_id_list[trainset_number:]
    if config['save_logging']:
        logging.info("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))
    print("{} datasets are used for training & {} datasets are used for testing".format(len(train_dataset), len(test_dataset)))


    # data manager init
    data_manager = DataManager(cache_path, dataset_name, data_root, model_name)
    # pulibc values on training datasets
    public_mean, public_std = get_public_value(cache_path, dataset_name, model_name, layer, train_dataset)
    data_manager.set_public_value(public_mean, public_std)

    for known_id in tqdm(train_dataset, total=len(train_dataset)):
        known_id = int(known_id)
        train_size = 1 - config['test_size']
        data_manager.add_dataset(known_id, layer, train_size, center=config['center'], scale=config['scale'], device=device, target_dir=args.target_acts)
    
    for known_id in tqdm(test_dataset, total=len(test_dataset)):
        if known_id in train_dataset:
            continue
        known_id = int(known_id)
        data_manager.add_dataset(known_id, layer, train_size=None, center=config['center'], scale=config['scale'], device=device, target_dir=args.target_acts)
    

    # training
    train_acts, train_labels = data_manager.get('train')
    probe, loss_list = LRProbe.train_dataset(train_acts, train_labels, lr=config['lr'], weight_decay=config['weight_decay'], epochs=config['epochs'], device=device, retain_loss=True)
    probe.set_public_value(public_mean, public_std)
    if config['save_logging']:
        logging.info(loss_list)

    # validation & iid
    cur_acts, cur_labels = data_manager.get('val')
    preds = probe.pred(cur_acts.to(device), iid=True)
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
        cur_acts, cur_labels = data_manager.get(known_id)
        preds = probe.pred(cur_acts.to(device), iid=False)
        cf_test.update(cur_labels, preds)
    if config['save_logging']:
        logging.info("Results of testset")
        logging.info(cf_test.__str__())
    deps.print_loud("no iid.\t" + cf_test.__str__())
        

    

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='main_exp.yaml')
    parser.add_argument("--model_name", type=str, default='gpt2-medium')
    parser.add_argument("--target_acts", type=str, default='hidden_status', choices=['hidden_status', 'attn'])
    
    # generalization settings (note that these parameters will be invalid under main setting)
    parser.add_argument("--mode", type=str, default='main', choices=['main', 'generalization', 'layer-total', 'layer-fewshot'])
    # for mode: generalization
    parser.add_argument('--train_dataset_number', type=int, default=1)
    # for mode: layer (note that main experiment's layer parameter is set in config file, not here)
    parser.add_argument('--layer_mode_layer', type=int, default=-1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_arg_parser()
    config_path = os.path.join('config', args.model_name, args.config_file)
    config = deps.load_yaml(config_path)

    assert args.model_name == config['model_name']
    
    if 'layer' not in args.mode and args.layer_mode_layer != -1:
        assert False, 'the parameter layer_mode_layer should not be used in other modes except the mode layer'
    

    deps.set_seed(config['seed'])
    current_time = deps.get_cur_time()
    config['current_time'] = current_time

    config['mode'] = args.mode
    config['train_dataset_number'] = args.train_dataset_number

    if config['save_logging']:

        if args.target_acts == 'hidden_status':
            ours_dir = 'ours'
        elif args.target_acts == 'attn':
            ours_dir = 'ours-attn'
        
        if args.mode == 'generalization':
            ours_dir = os.path.join('generalization', ours_dir, 'train_dataset_number_{}'.format(args.train_dataset_number))
        if 'layer' in args.mode:
            ours_dir = os.path.join('layer_mode', ours_dir + '-' + args.mode)
            config['layer'] = args.layer_mode_layer
        
        if args.config_file == 'main_exp.yaml':
            logging_path = os.path.join(config['cache_path'], 'logs', config['model_name'], ours_dir, '[{}]'.format(str(config['layer'])), current_time + '/')
        else:
            logging_path = os.path.join(config['cache_path'], 'logs', config['dataset_name'], config['model_name'], ours_dir, '[{}]'.format(str(config['layer'])), current_time + '/')
        deps.create_folders(logging_path)
        config['logging_path'] = logging_path
        logging.basicConfig(filename=os.path.join(logging_path, 'logging.log'), filemode='w', format='%(asctime)s %(name)s:%(levelname)s:%(message)s', datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        logging.info(config)

        deps.save_yaml(os.path.join(logging_path, 'main_exp.yaml'), config)        
        deps.print_loud("The log is saved in path {}".format(logging_path))

    main(config)