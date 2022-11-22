import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
torch.set_grad_enabled(False)


class AttentionMapExperiment():
    def __init__(self, model_name, data):
        self.available_models = [
            "sberbank-ai/ruRoberta-large",
            "cointegrated/rubert-tiny2",
            "DeepPavlov/rubert-base-cased",
            "sberbank-ai/ruBert-large"
        ]

        self.model_name = model_name

        self.model = AutoModel.from_pretrained(self.model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.size = (self.model.config.num_hidden_layers,
                     self.model.config.num_attention_heads)

        self.data = pd.read_csv(data)

        self.unique_labels = np.unique(self.data["role"]).tolist()
        self.indices = {u: self.data[self.data["role"] == u].index for u in self.unique_labels}
        self.data["role_idx"] = self.data["role"].apply(lambda x: self.unique_labels.index(x))

        self.attentions = None
        self.role_peaks = None

    def get_attention(self, sent_text, target_offset, head_offset):
        inputs = self.tokenizer.encode_plus(sent_text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs["input_ids"]
        offsets_mapping = inputs['offset_mapping']
        attention = self.model(input_ids)[-1]
        attention = self._tuple_of_tensors_to_tensor(attention)
        return self._get_attention(attention, offsets_mapping, target_offset, head_offset)

    def get_peaks(self, attentions):
        quantile = np.quantile(attentions, 0.75)
        peaks_indices = np.where(attentions >= quantile)
        attention_pos, peaks = np.unique(peaks_indices[1], return_counts=True)
        mask = np.zeros(attentions.shape[1])
        mask[attention_pos] = peaks / attentions.shape[0]
        return mask.reshape(self.size)

    def run_experiment(self):
        attention_data = []
        for i in tqdm(range(self.data.shape[0])):
            attention_data.append(self.get_attention(self.data.loc[i, "sentence"],
                                                     self.data.loc[i, "idx_target"],
                                                     self.data.loc[i, "idx_head"]))
        self.attentions = torch.stack(np.array(attention_data).tolist(), dim=0)

        role_peaks = {}
        for role in self.unique_labels:
            role_peaks[role] = self.get_peaks(self.attentions[self.indices[role], :])
        self.role_peaks = role_peaks

    def role_maps(self):
        if self.attentions is None:
            self.run_experiment()

        rows = len(self.unique_labels) // 2 + 1
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
        for ax, role in zip(np.ravel(axes), self.unique_labels):
            sns.heatmap(self.role_peaks[role], ax=ax)
            ax.set_title(role)
        plt.show()

    def role_similarity(self):
        top_attention_flat = {k: self.role_peaks[k].flatten() for k in self.role_peaks}
        cos_dist = cosine_distances(list(top_attention_flat.values()))
        matrix = np.triu(cos_dist)
        sns.heatmap(cos_dist, mask=matrix,
                    xticklabels=top_attention_flat.keys(), yticklabels=top_attention_flat.keys())
        plt.show()

    def role_prediction_accuracy(self):
        role_peaks_flat = np.array([self.role_peaks[k].flatten() for k in self.role_peaks])
        data_peaks = [self.get_peaks(self.attentions[i, :].unsqueeze(0)).flatten() for i in range(len(self.data))]
        data_peaks = np.stack(data_peaks, axis=0)
        cos_dist = cosine_distances(data_peaks, role_peaks_flat)
        pred = cos_dist.argmax(axis=1)
        accuracy = (pred == self.data["role_idx"]).sum() / self.data.shape[0]
        #print(f"Role prediction accuracy using attentions data for {self.model_name} is {round(accuracy, 4)}")
        return accuracy

    def _tuple_of_tensors_to_tensor(self, tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0).squeeze()

    def _text_to_slice(self, text_slice):
        return (int(item) for item in text_slice.split("-"))

    def word_to_tokens(self, token_offset, offsets_mapping):
        start, finish = self._text_to_slice(token_offset)
        start = (offsets_mapping[0, 1:-1, 0] == start).nonzero().item() + 1
        finish = (offsets_mapping[0, 1:-1, 1] == finish).nonzero().item() + 1
        return np.arange(start, finish + 1)

    def _get_attention(self, attention,
                       offsets_mapping,
                       target_offset,
                       head_offset):
        target_tokens = self.word_to_tokens(target_offset, offsets_mapping)
        head_tokens = self.word_to_tokens(head_offset, offsets_mapping)

        a = torch.mean(attention[:, :, head_tokens, :], dim=2)
        a = torch.sum(a[:, :, target_tokens], dim=2)
        return a.ravel()

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value not in self.available_models:
            raise Exception("This model is not supported")
        self._model_name = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        drop_idx = data[data["idx_head"].apply(lambda x: x.isalpha())].index
        data.drop(index=drop_idx, inplace=True)
        data.reset_index(drop=True, inplace=True)
        self._data = data