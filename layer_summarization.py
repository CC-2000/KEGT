import os
import argparse
import pandas as pd
import re

from utils import Log_Reader

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3.1-8b')
    parser.add_argument("--target_acts", type=str, default='hidden_status', choices=['hidden_status', 'attn'])
    parser.add_argument("--log_path", type=str, default="/home_sda1/wjy/cache/work2_try2/logs")
    parser.add_argument("--mode", default='layer-fewshot', choices=['layer-total', 'layer-fewshot'])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_arg_parser()
    
    if args.target_acts == 'hidden_status':
        ours_dir = 'ours'
    elif args.target_acts == 'attn':
        ours_dir = 'ours-attn'
    
    ours_dir = ours_dir + '-' + args.mode
    
    target_folder = os.path.join(args.log_path, args.model_name, "layer_mode", ours_dir)

    layer_list = os.listdir(target_folder)
    layer_list = list(map(lambda x: int(x[1:-1]), layer_list))
    layer_list = sorted(layer_list)

    print(layer_list)

    with open(os.path.join(target_folder, 'summarization.log'), 'w') as fp:
        fp.write(args.model_name + '\n')
        fp.write(args.target_acts + '\n')

        fp.write('\n')

        fp.write('\tlogs\taccuracy\tprecision\trecall\tf1 score\taccuracy\tprecision\trecall\tf1 score\n')
        for layer in layer_list:
            
            log_path = os.path.join(
                target_folder,
                '[{}]'.format(layer),
            )
            cur_time = os.listdir(log_path)[0]      # 第一个
            log_path = os.path.join(
                log_path,
                cur_time,
                'logging.log'
            )
        
            log_reader = Log_Reader(log_path)
            assert log_reader.cur_time() == cur_time

            fp.write(str(layer) + '\t')
            fp.write(log_reader.cur_time() + '\t')
            for match in log_reader.matches:
                accuracy, precision, recall, f1_score = match
                fp.write(accuracy + '\t' + precision + '\t' + recall + '\t' + f1_score + '\t')
            fp.write('\n')
        fp.write('\n')
