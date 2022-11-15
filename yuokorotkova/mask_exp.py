import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import pandas as pd
import pymorphy2
from transformers import pipeline

class MaskExperiment:
    def __init__(self, model_name, data, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.morph = pymorphy2.MorphAnalyzer()

        self.dataset = pd.read_csv(data)

        self.available_models = [
                        "sberbank-ai/ruRoberta-large", 
                        "google/mt5-small", 
                        "cointegrated/rubert-tiny2", 
                        "DeepPavlov/rubert-base-cased", 
                        "sberbank-ai/ruBert-large"
                        ]
        self.masks = {
            "sberbank-ai/RoBERTa-large": " <mask> ",
            "google/mt5-small": "<extra_id_0>",
            "cointegrated/rubert-tiny2": " [MASK] ",
            "DeepPavlov/rubert-base-cased": " [MASK] ",
            "sberbank-ai/ruBert-large": " [MASK] ",
        }

        if self.model_name == 'google/mt5-small':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            _ = self.model.to(self.device)
        elif self.model_name in self.available_models:
            self.unmasker = pipeline("fill-mask", model=self.model_name)
        else:
            raise Exception("Not supported model name")

    def run_experiment(self):
        self.dataset["masked_sentence"] = self.dataset.apply(lambda x: self.mask_sentence(x), axis=1)
        if self.model_name == 'google/mt5-small':
            self.dataset["arguments"] = self.dataset["masked_sentence"].apply(self.fill_t5)
        else:
            self.dataset["arguments"] = self.dataset["masked_sentence"].apply(self.fill_bert)
        self.dataset["score"] = self.dataset.apply(lambda x: self.parse_args(x), axis=1)
        return {
            "by role": self.dataset.groupby('role')["score"].mean(),
            "average": self.dataset["score"].mean(axis=0)
            }

    def fill_bert(self, text):
        return [i['token_str'] for i in self.unmasker(text, top_k=10)]

    def fill_t5(self, text, n=10, max_length=30, top_p=0.5):
        input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input["input_ids"].to(self.device),
                attention_mask=input["attention_mask"].to(self.device),
                repetition_penalty=10.0, 
                max_length=max_length, 
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=n,
                top_p=top_p,
            )
        results = []
        for o in out:
            r = re.search("(?<=>).+?(?=<)", self.tokenizer.decode(o, skip_special_tokens=True))
            if r:
                results.append(r.group())
        return list(set(results))
    
    def mask_sentence(self, row):
        start, end = row["idx_target"].split("-")
        start, end = int(start), int(end)
        sentence = row["sentence"]
        return sentence[:start] + self.masks[self.model_name] + sentence[end:]

    def parse_args(self, row):
        args = row["arguments"]
        case = self.morph.parse(row["raw_target"])[0].tag.case
        tokens = []
        correct = 0
        all_tokens = 0
        for arg in args:
            tokens.extend(re.findall("\w+", arg))
        if not tokens:
            return None
        for arg in tokens:
            parses = self.morph.parse(arg)
            for p in parses:
                if p.tag.POS in ["NPRO", "NOUN"]:
                    if p.tag.case == case:
                        correct += 1
                        all_tokens += 1
                        break
                    else:
                        all_tokens += 1
                        break
        return correct / len(tokens)