import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manager import Wiki_Data_Manager
import nltk
import pandas as pd
import random
from sklearn.manifold  import TSNE

from utils import plot, get_pcs
import deps
import argparse




def get_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=20001202)

    parser.add_argument("--model_hub_path", type=str, default='/home_sda1/wjy/model_hub')
    parser.add_argument("--model_name", type=str, default='llama3.1-8b')
    parser.add_argument("--device_id", type=int, default=0)

    parser.add_argument("--data_root", type=str, default='./data/NQ/')

    parser.add_argument("--cache_path", type=str, default='/home_sda1/wjy/cache')
    parser.add_argument("--dataset_name", type=str, default='NQ')

    parser.add_argument("--other_datasets", default="NQ", choices=['KASS', "NQ", "WebQ", "Amazon"])
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":


    args = get_arg_parser()
    deps.set_seed(args.seed)
    device = deps.get_device(args.device_id)

    model, tok = deps.load_model(args.model_hub_path, args.model_name, device)

    print(model)

    wiki_data = Wiki_Data_Manager()

    obj_alias = wiki_data.obj2alias
    entity_list = []
    for obj_index in obj_alias.keys():
        entity_list += obj_alias[obj_index]



    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = stop_words + ['\'s', '.', ',', '!', '?', '\'', '\"', ' ', '', '(', ')', '[', ']', '{', '}', ':', ';', '-', '_', '=', '+', '*', '/', '\\', '|', '<', '>', '@', '#', '$', '%', '^', '&', '~', '`', '``'] 
    english_words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    number_words = ['1', '2' ,'3', '4', '5', '6', '7', '8', '9', '0']
    stop_words = stop_words + english_words
    stop_words = stop_words + number_words

    statements_template = 'For the prompt "{}", the token "{}" will appear in the generated text for the prompt above.'

    statements_batch = 1

    prompts = []
    fact_ids = []

    if args.other_datasets == 'KASS':
        for i in range(len(wiki_data.data)):
            prompts = prompts + wiki_data[i][1]
            fact_ids = fact_ids + [wiki_data[i][0]] * len(wiki_data[i][1])
    else:
        # load other datasets
        dataset_file_path = os.path.join('./data/', args.other_datasets, args.other_datasets + '.json')
        other_data = deps.json_data_load(dataset_file_path)
        for d in other_data:
            prompts.append(d['prompt'])
            fact_ids.append(d['fact_id'])


    df = pd.DataFrame(columns=['known_id', 'statement', 'token', 'prompt', 'correct_token', 'correct_token_index', 'label'])

    for i in tqdm(range(len(prompts)), total=len(prompts)):    # range(len(prompts))

        prompt = prompts[i]
        fact_id = fact_ids[i]

        _, generate_txt = deps.generate_next_k(model, tok, [prompt], 10)
        
        idx_list = random.sample(list(range(len(generate_txt))), len(generate_txt))

        flag = True
        for idx in idx_list:
            if generate_txt[idx].strip().lower() in stop_words:
                continue
            flag = False
            
            neg_obj = random.sample(entity_list, k=1)[0]

            correct_token = generate_txt[idx]
            wrong_token = neg_obj

            statements_correct = statements_template.format(prompt, correct_token)
            statements_wrong = statements_template.format(prompt, wrong_token)
        
            df.loc[len(df)] = [fact_id, statements_correct, correct_token, prompt, correct_token, idx + 1, 1]
            df.loc[len(df)] = [fact_id, statements_wrong, wrong_token, prompt, correct_token, idx + 1, 0]

            break
        
        if flag is True:
            df.loc[len(df)] = [fact_id, statements_correct, correct_token, prompt, correct_token, idx + 1, 1]
            df.loc[len(df)] = [fact_id, statements_wrong, wrong_token, prompt, correct_token, idx + 1, 0]
    if args.other_datasets is None:
        df.to_csv('./data/KASS/KASS_{}.csv'.format(args.model_name), index=None)
    else:
        other_dataset_save_path = os.path.join('./data/', args.other_datasets, args.other_datasets + '_{}.csv'.format(args.model_name))
        df.to_csv(other_dataset_save_path, index=None)