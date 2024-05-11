import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import gc 

import pickle

import faiss

from datasets import load_dataset
import datasets

from transformers import AutoTokenizer, AutoModel
import torch

wiki_dumps = []
num_of_dumps = 4
dump_names = "dumps_1/wiki_dump_out_"

for i in range(num_of_dumps):
    dbfile = open(f'{dump_names}{i}.pickle', 'rb')
    wiki_dumps.extend(pickle.load(dbfile))
    dbfile.close()

print("dump size : ", len(wiki_dumps))

wiki_db = datasets.Dataset.from_list(wiki_dumps)

model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("model")
device = torch.device('cuda:0')
model.to(device)


# split sections into chunks
MAX_TOKENS = 300
wikipedia_strings = []

def add_token_lens(x):

    token_checkpoint = 0
    collect_chunks = []
    collect_chunk_lens = []

    for i, char in enumerate(x['text']):
        if char == '.':
            tokens = tokenizer(
                [x['text'][token_checkpoint: i + 1]], padding = True, truncation = False, return_tensors = "pt"
            )
            tokens_len = tokens['input_ids'].shape[1]
            if tokens_len > 300:
                collect_chunks.append( (token_checkpoint, i + 1) )
                collect_chunk_lens.append( tokens_len )
                token_checkpoint = i + 1

    if token_checkpoint < len(x['text']):
        tokens = tokenizer(
                x['text'][token_checkpoint: len(x['text'])], padding = True, truncation = False, return_tensors = "pt"
        )
        tokens_len = tokens['input_ids'].shape[1]
        collect_chunks.append( (token_checkpoint,  len(x['text'])) )
        collect_chunk_lens.append( tokens_len )

    tokens = tokenizer(
        [x['text'][i:j] for i, j in collect_chunks],
        padding = True, truncation = True, return_tensors = "pt"
    )
    x['text'] = [x['text'][i:j] for i, j in collect_chunks]
    x['tokenized_chunks'] = tokens
    x['tokenized_chunks_lens'] = collect_chunk_lens
    return x

wiki_db = wiki_db.filter(lambda x: len(x['text']) > 100)

wiki_db = wiki_db.map(add_token_lens)
print(wiki_db)

wiki_db_extended = []
_id = 0

for x in wiki_db:
  for i, tokens_len in enumerate(x['tokenized_chunks_lens']):
    wiki_db_extended.extend([{
      '_id': _id,
      'title': x['title'],
      'site': x['site'],
      'text': x['text'][i],
      'tokens': {key: [value[i]] for key, value in x['tokenized_chunks'].items()},
      'tokens_len': tokens_len
    }])
    _id += 1

wiki_db_flattened = datasets.Dataset.from_list(wiki_db_extended)

del wiki_db
del wiki_db_extended
gc.collect()

print(wiki_db_flattened)

idx = 0

print("\n title : \n")
print(wiki_db_flattened[idx]['title'])
print("\n site : \n")
print(wiki_db_flattened[idx]['site'])
print("\n text : \n")
print(wiki_db_flattened[idx]['text'])
print("\n tokens_len : \n")
print(wiki_db_flattened[idx]['tokens_len'])
print("\n tokens : \n")
print(wiki_db_flattened[idx]['tokens'])


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    # print("embedding_input : ")
    # print(encoded_input)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def map_embeddings(chunks):
    encoded_input = {k: torch.tensor(v).to(device) for k, v in chunks.items()}
    embeddings = model(**encoded_input).last_hidden_state[:,0]
    return embeddings.detach().cpu().numpy()[0]

wiki_db_embedded = wiki_db_flattened.map(lambda x: {'embeddings' : map_embeddings(x['tokens'])})
print(wiki_db_embedded)


wiki_db_embedded.save_to_disk("wiki_db_embedded.hf")

wiki_db_embedded.add_faiss_index(column = "embeddings")

import numpy as np

query = 'cholesterol with fast food consumptions'

query_embedding = get_embeddings(query).detach().cpu()
query_embedding = np.asarray(query_embedding)

print(query_embedding.shape)

scores, samples = wiki_db_embedded.get_nearest_examples(
        "embeddings", query_embedding, k = 10
)

print(scores)
print(samples['title'])
print(samples['text'])




