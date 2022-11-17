import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import scipy.cluster.hierarchy as shc

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

    return linkage_matrix

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class BertTopicSLab():
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
        self.model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

        self.umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
        
        with torch.no_grad():
            self.model_output = self.model(**encoded_input)

        self.sentence_embeddings = mean_pooling(self.model_output, encoded_input['attention_mask'])
        self.umap_embeddings = UMAP(n_neighbors=15,
                                    n_components=5,
                                    min_dist=0.0,
                                    metric='cosine').fit(self.sentence_embeddings).transform(self.sentence_embeddings)


    def get_agglomerative_clusters(self):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(self.umap_embeddings)
        self.hierarchical_linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig("hierarchical.png", format="PNG", figsize=(10, 10))
        self.hierarchical_distance = model.distances_

    def get_basic_dendrogram(self):
        clusters = shc.linkage(self.umap_embeddings, method='ward', metric="euclidean")
        shc.dendrogram(Z=clusters)
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig("basic_dendrogram.png", format="PNG", figsize=(50, 50))
        plt.show()    