import torch
import torch.nn as nn
import numpy as np
import random
from tensorboardX import SummaryWriter
import os
import time
from nnsight import LanguageModel
from typing import Dict, Iterable, Callable
from torch import Tensor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    print("|{:^15}|{:^15}|{:^15}|".format("", "one", "billion"))
    tmp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("|{:^15}|{:^15}|{:^15}|".format("training params", tmp, format(tmp / 1e9, '.2f')))
    tmp = sum(p.numel() for p in model.parameters())
    print("|{:^15}|{:^15}|{:^15}|".format("all params", tmp, format(tmp / 1e9, '.2f')))


def get_device(cuda_number: int):
    if cuda_number >= 0:
        device = torch.device(f'cuda:{cuda_number}')
    else:
        device = torch.device('cpu')
    print(f"Running task on device {device}")
    return device

def get_tensor_writer(mode='current_time'):
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    if mode == 'current_time':
        return SummaryWriter(log_dir=os.path.join('logs', current_time)), current_time
    else:
        raise NotImplemented


def collect_hidden_states(prompts, model, tok, layers, target_token_idx=-1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    sight_model = LanguageModel(model, tokenizer=tok)
    acts = {}
    with sight_model.trace(prompts) as tracer:
        for layer in layers:
            acts[layer] = sight_model.model.layers[layer].output[0][:,target_token_idx,:].save()

    for layer, act in acts.items():
        acts[layer] = act.value
    
    return acts

def collect_hiiden_states_no_nnsight(prompts, model, tok, layers, target_token_idx=-1):
    assert len(prompts) == 1, "Not supporting simultaneous inference of multiple prompts"
    acts = {}
    inps = tok(prompts, return_tensors='pt').to(model.device)
    outs = model(input_ids=inps['input_ids'], attention_mask=inps['attention_mask'], output_hidden_states=True)
    hidden_states = outs.hidden_states

    for layer in layers:
        acts[layer] = hidden_states[layer + 1][:, target_token_idx, :] # hidden_states的第0个是embedding layer's output

    return acts


class FeatureExtractor():
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        self.removable_handle = []

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            handle = layer.register_forward_hook(self.save_outputs_hook(layer_id))
            self.removable_handle.append(handle)

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def collect_features(self):
        assert len(self.removable_handle) != 0
        return self._features

    def remove_handle(self):
        for handle in self.removable_handle:
            handle.remove()
        self.removable_handle = []


def collect_features_no_nnsight(prompts, model, tok, decoder_module_name, layers, target_token_idx=-1):
    """
        decoder_module_name: (for llama3.1)
            model.layers.{}.self_attn
            or
            model.layers.{}.mlp
    """
    assert len(prompts) == 1, "Not supporting simultaneous inference of multiple prompts"
    acts = {}

    named_modules = [decoder_module_name.format(layer) for layer in layers]

    fe = FeatureExtractor(model, named_modules)

    inps = tok(prompts, return_tensors='pt').to(model.device)
    outs = model(input_ids=inps['input_ids'], attention_mask=inps['attention_mask'])
    # hidden_states = outs.hidden_states
    collected_features = fe.collect_features()

    for layer in layers:
        layer_name = decoder_module_name.format(layer)
        acts[layer] = collected_features[layer_name][0][:, target_token_idx, :] # hidden_states的第0个是embedding layer's output

    fe.remove_handle()
    return acts


if __name__ == '__main__':

    class Model(nn.Module):
        def __init__(self, num_inputs, num_hiddens, num_outputs):
            super().__init__()
            self.hidden = nn.Linear(num_inputs, num_hiddens)
            self.act = nn.ReLU()
            self.output = nn.Linear(num_hiddens, num_outputs)

        def forward(self, x):
            return self.output(self.act(self.hidden(x)))
    

    num_inputs = 100
    num_hiddens = 200
    num_outputs = 2
    num_ins = 2

    net = Model(num_hiddens=num_hiddens, num_outputs=num_outputs, num_inputs=num_inputs)
    net.hidden.requires_grad_(False)


    count_params(net)