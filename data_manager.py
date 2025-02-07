import os
import argparse
from pprint import pprint
from tqdm import tqdm
import random
import pandas as pd
import torch

from utils import load_acts
import numpy as np
import deps
import json
import nltk
import random

class Wiki_Data_Manager():

    def __init__(self, base_dir='./data/chen'):
        with open(os.path.join(base_dir, 'allsub2alias.json')) as fp:
            self.sub2alias = json.load(fp)
        with open(os.path.join(base_dir, 'allobj2alias.json')) as fp:
            self.obj2alias = json.load(fp)
        with open(os.path.join(base_dir, 'relation2template.json')) as fp:
            self.relation2template = json.load(fp)
        self.data = deps.json_data_load(os.path.join(base_dir, 'chen_upper_10_and_lower_30_new.json'))
    
    def __getitem__(self, index):
        rel = self.data[index]['relation']
        sub = self.data[index]['subj_label']
        obj = self.data[index]['obj_label']
        fact_id = self.data[index]['fact_id']
        
        prompts = self.prepare(self.sub2alias[sub], self.relation2template[rel])

        return fact_id, prompts

    def prepare(self, subAlias, relTemplate):
        prompts = []
        for s in subAlias:
            for r in relTemplate:
                prompts.append(r.replace('[X]', s).replace('[Y].', '')[:-1])
        return prompts 
    
    def mode_standard(self):
        self.all_prompts = []
        for i in range(len(self.data)):
            self.all_prompts = self.all_prompts + self[i][1]
        

class DataManager():

    def __init__(self, cache_dir, outer_dataset_name, outer_data_path, model_name):
        
        self.cache_dir = cache_dir
        self.outer_dataset_name = outer_dataset_name
        self.model_name = model_name
        self.df = pd.read_csv(os.path.join(outer_data_path, outer_dataset_name + '_' + model_name + '.csv'))
        self.group_df = self.df.groupby('known_id')

        self.data = {
            'train': {},
            'val': {},
        }

        self.public_mean = None
        self.public_std = None
    
    def set_public_value(self, p_m, p_s):
        self.public_mean = p_m
        self.public_std = p_s
        
    
    def add_dataset(self, known_id, layer, train_size=None, center=True, scale=False, device='cpu', target_dir='hidden_status'):
        acts = load_acts(self.cache_dir, self.outer_dataset_name, self.model_name, known_id, layer, target_dir)
        
        if self.public_mean is not None:
            cur_mean = self.public_mean
            cur_std = self.public_std
        else:
            cur_mean = torch.mean(acts, dim=0)
            cur_std = torch.std(acts, dim=0)

        if center:
            acts = acts - cur_mean
        if scale:
            acts = acts / cur_std
            
        cur_df = self.group_df.get_group(known_id).copy()
        labels = torch.Tensor(cur_df['label'].values).to(device)

        if train_size is None:
            self.data[known_id] = acts, labels
        else:
            assert 0 < train_size and train_size < 1
            
            train = torch.randperm(len(cur_df)) < int(train_size * len(cur_df))
            val = ~train

            self.data['train'][known_id] = acts[train], labels[train]
            self.data['val'][known_id] = acts[val], labels[val]
    

    def get(self, dataset):
        if dataset == 'train':
            data_dict = self.data['train']
        elif dataset == 'val':
            data_dict = self.data['val']
        elif isinstance(dataset, int):
            data_dict = {dataset: self.data[dataset]}            
        else:
            raise ValueError("NOT IMPLEMENT")


        all_cats, all_labels = [], []
        for known_id in data_dict:
            acts, labels = data_dict[known_id]
            all_cats.append(acts)
            all_labels.append(labels)
        return torch.cat(all_cats, dim=0), torch.cat(all_labels, dim=0)
    


    

class CCS_DataManager:

    def __init__(self, cache_dir, outer_dataset_name, outer_data_path, model_name):
        self.cache_dir = cache_dir
        self.outer_dataset_name = outer_dataset_name
        self.model_name = model_name
        self.df = pd.read_csv(os.path.join(outer_data_path, outer_dataset_name + '_' + model_name + '.csv'))
        self.group_df = self.df.groupby('known_id')

        self.data = {
            'train': {},
            'val': {},
        }

        self.public_mean = None
        self.public_std = None
    
    def format_prompt_1(self, text, label):
        return "The following statement is " + ["incorrect", "correct"][label] + ":\n" + text


    def get_hidden_states_many_examples(self, model, tok, df, layer):
        """
        Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
        Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
        with the ground truth labels
        
        This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
        """
        data = df
        # setup
        model.eval()
        all_neg_hs, all_pos_hs, all_gt_labels = [], [], []


        for i in tqdm(range(len(data)), total=len(data)):
            cur_data = data.iloc[i]

            # ccs_prompt = format_prompt_1(cur_data['statement'], cur_data['label'])
            # ccs_prompt = format_prompt_2(cur_data['prompt'], cur_data['token'], cur_data['label'])


            # pos_css_prompt = format_prompt_2(cur_data['prompt'], cur_data['token'], pos_label)
            # neg_css_prompt = format_prompt_2(cur_data['prompt'], cur_data['token'], neg_label)
            pos_css_prompt = self.format_prompt_1(cur_data['statement'], 1)
            neg_css_prompt = self.format_prompt_1(cur_data['statement'], 0)

            pos_hs = deps.collect_hiiden_states_no_nnsight([pos_css_prompt], model, tok, [layer])[layer].detach().cpu()
            neg_hs = deps.collect_hiiden_states_no_nnsight([neg_css_prompt], model, tok, [layer])[layer].detach().cpu()

            all_neg_hs.append(neg_hs)
            all_pos_hs.append(pos_hs)
            all_gt_labels.append(cur_data['label']) # 也就是cur_data['label']


        all_neg_hs = np.stack(all_neg_hs)
        all_pos_hs = np.stack(all_pos_hs)
        all_gt_labels = np.stack(all_gt_labels)

        return torch.tensor(all_neg_hs[:, -1, :]), torch.tensor(all_pos_hs[:, -1, :]), torch.tensor(all_gt_labels)
    

    def add_dataset(self, model, tok, known_id, layer, train_size=None, device='cpu', target_dir='hidden_status'):

        # acts = load_acts(self.cache_dir, self.outer_dataset_name, self.model_name, known_id, layer, target_dir)
    
        cur_df = self.group_df.get_group(known_id).copy()
        # labels = torch.Tensor(cur_df['label'].values).to(device)
        neg_hs, pos_hs, labels = self.get_hidden_states_many_examples(model, tok, cur_df, layer)

        if train_size is None:
            self.data[known_id] = pos_hs, neg_hs, labels
        else:
            assert 0 < train_size and train_size < 1
            
            train = torch.randperm(len(cur_df)) < int(train_size * len(cur_df))
            val = ~train

            self.data['train'][known_id] = pos_hs[train], neg_hs[train], labels[train] # acts[train], labels[train]
            self.data['val'][known_id] = pos_hs[val], neg_hs[val], labels[val]   #acts[val], labels[val]
    

    def get(self, dataset):
        if dataset == 'train':
            data_dict = self.data['train']
        elif dataset == 'val':
            data_dict = self.data['val']
        elif isinstance(dataset, int):
            data_dict = {dataset: self.data[dataset]}            
        else:
            raise ValueError("NOT IMPLEMENT")


        all_pos_acts, all_neg_acts, all_labels = [], [], []
        for known_id in data_dict:
            pos_hs, neg_hs, labels = data_dict[known_id]
            all_pos_acts.append(pos_hs)
            all_neg_acts.append(neg_hs)
            all_labels.append(labels)

        return torch.cat(all_pos_acts, dim=0).numpy(), torch.cat(all_neg_acts, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()
    