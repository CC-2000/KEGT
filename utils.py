import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, OPTForCausalLM
from transformers import AutoTokenizer, AutoModel
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import re

from deps import top_k_sampling



    
def get_pcs(X, k=2, offset=0):
    """
    Performs Principal Component Analysis (PCA) on the n x d data matrix X. 
    Returns the k principal components, the corresponding eigenvalues and the projected data.
    """

    # Subtract the mean to center the data
    X = X - torch.mean(X, dim=0)
    
    # Compute the covariance matrix
    cov_mat = torch.mm(X.t(), X) / (X.size(0) - 1)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_mat)
    
    # Since the eigenvalues and vectors are not necessarily sorted, we do that now
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the pcs
    eigenvectors = eigenvectors[:, offset:offset+k]
    
    return eigenvectors


def load_acts(cache_dir, dataset_name, model_name, known_id, layer, target_dir='hidden_status'):
    dir_path = os.path.join(cache_dir, target_dir, model_name, dataset_name, str(known_id))
    file_name = f"layer_{layer}.pt"
    hs = torch.load(os.path.join(dir_path, file_name))
    return hs




def plot(df, dimensions, dim_offset=0, arrows=[], return_df=False, **kwargs):
    
    acts = df['activation'].tolist()
    acts = torch.stack(acts, dim=0).cuda()
    pcs = get_pcs(acts, dimensions, offset=dim_offset)

    acts = df['activation'].tolist()
    acts = torch.stack(acts, dim=0).cuda()
    proj = torch.mm(acts, pcs)

    # add projected data to df
    for dim in range(dimensions):
        df[f"PC{dim+1}"] = proj[:, dim].tolist()
    
    # shuffle rows of df
    df = df.sample(frac=1)
    
    # plot using plotly
    if dimensions == 2:
        fig = px.scatter(df, x='PC1', y='PC2', 
                            hover_name='statement', 
                            color_continuous_scale='Bluered_r',
                            **kwargs)
    elif dimensions == 3:
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', 
                            hover_name='statement', 
                            color_continuous_scale='Bluered_r',
                            **kwargs)
    else:
        raise ValueError("Dimensions must be 2 or 3")

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    
    fig.update_layout(
        coloraxis_showscale=False,
    )
    
    if arrows != []:
        for i, arrow in enumerate(arrows): # arrow is a tensor of shape [acts.shape[1]]
            arrow = arrow.to(pcs.device)
            arrow = torch.mm(arrow.unsqueeze(0), pcs)
            arrow = go.layout.Annotation(
                x=arrow[0,0],
                y=arrow[0,1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                ax=0,
                ay=0,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                opacity=0.8,
                showarrow=True,
            )
            arrows[i] = arrow
        
        fig.update_layout(annotations=arrows)

    if return_df:
        return fig, df
    else:
        return fig
        


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
    


class ConfusionMatrix:
    
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
    
    def add(self, tp, fp, tn, fn):
        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn
    
    def update(self, true_label, pred_label):
        temp = true_label == pred_label
        self.TP += temp[true_label == 1].sum()
        self.TN += temp[true_label == 0].sum()
        self.FP += ((pred_label - true_label) == 1).sum()
        self.FN += ((pred_label - true_label) == -1).sum()
    
    def addConfusionMatrix(self, confusionMatrix):
        self.TP += confusionMatrix.TP
        self.FP += confusionMatrix.FP
        self.TN += confusionMatrix.TN
        self.FN += confusionMatrix.FN
    
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    
    def precision(self):
        return self.TP / (self.TP + self.FP)
    
    def recall(self):
        return self.TP / (self.TP + self.FN)

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        accuracy = self.accuracy()
        f1_score = 2 * precision * recall / (precision + recall)
        return f'accuracy: {accuracy}\tprecision: {precision}\trecall: {recall}\tf1 score: {f1_score}'
        

class Log_Reader:

    def __init__(self, log_file_path):
        with open(log_file_path, 'r') as fp:
            self.log_content = fp.read()

        self.metrics_pattern = r'accuracy:\s*(\d+\.\d+)\s*precision:\s*(\d+\.\d+)\s*recall:\s*(\d+\.\d+)\s*f1 score:\s*(\d+\.\d+)'
        self.matches = re.findall(self.metrics_pattern, self.log_content)
        
        current_time_pattern = r"'current_time': '([^']+)'"
        self.current_time = re.findall(current_time_pattern, self.log_content)[0]
    
    def cur_time(self):
        return self.current_time

    def get_matches(self):
        return self.matches