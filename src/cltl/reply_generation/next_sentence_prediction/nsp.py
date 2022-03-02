""" Filename:     NSP.py
    Author(s):    Thomas Bellucci
    Description:  Implementation of a BERT-based next sentence prediction
                  (NSP) transformer built on Huggingface transformers.
    Date created: Nov. 11th, 2021
"""

import numpy as np
import torch
from transformers import (BertForNextSentencePrediction,
                          BertTokenizer)

from cltl.reply_generation.api import ThoughtSelector


class NSP(ThoughtSelector):
    def __init__(self, filename):
        """Initializes an instance of BERT for Next Sentence Prediction (NSP).

        params
        str filename: path to a pretrained NSP BERT model

        returns: None
        """
        self.__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.__model = BertForNextSentencePrediction.from_pretrained(filename)

        self.__device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.__model.to(self.__device)

    def score_response(self, context, response):
        """Predicts for a (context, response) pair their likelihood according to
        the model.

        returns: Softmax likelihood
        """
        X_batch = self.__tokenizer.batch_encode_plus(
            [[context, response]], padding=True, truncation=True, return_tensors="pt"
        )

        X_batch["input_ids"] = X_batch["input_ids"].to(self.__device)
        X_batch["token_type_ids"] = X_batch["token_type_ids"].to(self.__device)
        X_batch["attention_mask"] = X_batch["attention_mask"].to(self.__device)

        # Forward pass
        outputs = self.__model(**X_batch)
        logits = outputs.logits.detach().cpu().numpy()[0]

        # Prob(is_next) using softmax
        return np.exp(logits[0]) / np.sum(np.exp(logits))

    def select(self, scores):
        scores.sort(key=lambda x: x[2], reverse=True)

        return scores[0]
