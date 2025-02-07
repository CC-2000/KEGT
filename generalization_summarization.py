import os
import argparse
import pandas as pd
import re

from utils import Log_Reader


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3.1-8b')
    parser.add_argument("--target_acts", type=str, default='hidden_status', choices=['hidden_status', 'attn', 'ccs'])
    parser.add_argument("--log_path", type=str, default="/home_sda1/wjy/cache/work2_try2/logs")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_arg_parser()
    
    if args.target_acts == 'hidden_status':
        ours_dir = 'ours'
    elif args.target_acts == 'attn':
        ours_dir = 'ours-attn'
    elif args.target_acts == 'ccs':
        ours_dir = 'ccs'
    target_folder = os.path.join(args.log_path, args.model_name, "generalization", ours_dir)

    train_number_list = os.listdir(target_folder)
    train_number_list = list(map(lambda x: int(x.split('_')[-1]), train_number_list))
    layer_list = os.listdir(os.path.join(target_folder, 'train_dataset_number_{}'.format(train_number_list[0])))
    layer_list = list(map(lambda x: int(x[1:-1]), layer_list))

    train_number_list = sorted(train_number_list)
    layer_list = sorted(layer_list)

    print(train_number_list)
    print(layer_list)

    with open(os.path.join(target_folder, 'summarization.log'), 'w') as fp:
        fp.write(args.model_name + '\n')
        fp.write(args.target_acts + '\n')

        fp.write('\n')

        for layer in layer_list:
            fp.write('layer ' + str(layer) + '\t')
            
            fp.write('logs\taccuracy\tprecision\trecall\tf1 score\taccuracy\tprecision\trecall\tf1 score\n')
            
            for train_number in train_number_list:
                log_path = os.path.join(
                    target_folder,
                    'train_dataset_number_{}'.format(train_number),
                    '[{}]'.format(layer),
                )
                cur_time = os.listdir(log_path)[0]      # 第一个
                log_path = os.path.join(
                    log_path,
                    cur_time,
                    'logging.log'
                )
            
                log_reader = Log_Reader(log_path)
                assert log_reader.cur_time() == cur_time, print(train_number, layer, cur_time, log_reader.cur_time())
                fp.write(str(train_number) + '\t')
                fp.write(log_reader.cur_time() + '\t')
                for match in log_reader.matches:
                    accuracy, precision, recall, f1_score = match
                    fp.write(accuracy + '\t' + precision + '\t' + recall + '\t' + f1_score + '\t')
                fp.write('\n')
            fp.write('\n')
