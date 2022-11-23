import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd

def run_experiment(df, model_name, labels):
    sentences = []
    for i in range(len(df['raw_target'])):
        out = ''
        out += str(df['raw_target'][i])
        out += ' '
        out += str(df['raw_head'][i]) 
        sentences.append(out)
    
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)  
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sents = [sent for sent in model_output[0]]
    s_ = [s.cpu().detach().numpy() for s in sents]
    ss = [s.flatten() for s in s_]
    clusters = shc.linkage(ss, method='ward', metric="euclidean")
    mergings = linkage(clusters, method='complete')
    plt.figure(figsize=(150, 150))
    dendrogram(mergings,
               labels=labels,
               leaf_rotation=90,
               leaf_font_size=6,
               );
    plt.show()