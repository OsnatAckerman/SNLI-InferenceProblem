STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import json
import spacy
import torch
from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab
import os
import sys
import nltk


cud = torch.cuda.is_available()
device = torch.device("cuda" if cud else "cpu")


def load_snlidata_json(text_field, label_field):
    train, dev, test = datasets.SNLI.splits(text_field, label_field)
    return train, dev, test


def read_data(batch_size):
    TEXT_FIELD = data.Field(sequential=True, preprocessing=lambda seq: ['NULL'] + seq,
                            batch_first=True, include_lengths=True, tokenize=data.get_tokenizer('spacy'))
    LABEL_FIELD = data.Field(sequential=False, batch_first=True, unk_token=None)

    snli_train, snli_dev, snli_test = load_snlidata_json(TEXT_FIELD, LABEL_FIELD)

    TEXT_FIELD.build_vocab(snli_train, snli_dev, snli_test,
                           vectors=vocab.GloVe(name='840B', dim=300))
    LABEL_FIELD.build_vocab(snli_train, snli_dev, snli_test)

    snli_train_iter, snli_val_iter, snli_test_iter = data.BucketIterator.splits(
        datasets=(snli_train, snli_dev, snli_test),
        batch_size=batch_size,
        repeat=False,
        sort_key=lambda x: len(x.premise), device=device)
    return (snli_train_iter, snli_val_iter, snli_test_iter), TEXT_FIELD, LABEL_FIELD


if __name__ == "__main__":

    (snli_train_iter, snli_val_iter, snli_test_iter), TEXT_FIELD, LABEL_FIELD = read_data(4)
    for batch in snli_train_iter:
        batch = batch
        xp, _ = batch.premise
        xh, _ = batch.hypothesis
        y = batch.label
        xp, xh, y = xp.to(device), xh.to(device), y.to(device)
        print(xp)
        print(xh)
        print(y)
