"""
Vector space models with ElasticSearch
"""

__author__ = "rawatshaurya1994@gmail.com"

import os
from elasticsearch import Elasticsearch
import tensorflow as tf
import tensorflow_hub as hub
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_PATH = os.path.join("app", "data")
DB_HOST = '127.0.0.1'
DB_PORT = 9200

# Encoder: Universal Sentence Encoder
ENCODER_PATH = os.path.join("app", "embeddings", "universal-sentence-encoder-large_5")
encoder = hub.load(ENCODER_PATH)


# Get data from files into Dataframe
def get_data(path):
    texts = []
    for doc in os.listdir(path):
        text = open(os.path.join(path, doc), "r", encoding='UTF-8').read()
        texts.append(text)
    return pd.DataFrame(texts)


# Setup ElasticSearch and create Index and Mapping
def es_setup(host, port):
    db = Elasticsearch([{'host': host, 'port': port}])
    if db.ping():
        print("Connected to ElasticSearch!")
    else:
        print("Error connecting to ElasticSearch")
    mapping = {"mappings": {
        "properties": {
            "document": {
                "type": "text"
            },
            "document_vector": {
                "type": "dense_vector",
                "dims": 512
            }
        }
    }}
    response = db.indices.create(
        index='documents',
        ignore=400,
        body=mapping
    )
    print(json.dumps(response, indent=5))


es_setup(DB_HOST, DB_PORT)


# Connection instance for ElasticSearch
def es_connect(host, port):
    db = Elasticsearch([{'host': host, 'port': port}])
    if db.ping():
        print("Connected!")
    else:
        print("Unable to connect")
    return db


# Index into ELasticSearch
def es_insert(db: Elasticsearch(), text_id: int, text: str, text_vector: list()):
    mapping = {
        "document": text,
        "document_vector": text_vector
    }
    db.index(
        index="documents",
        id=text_id,
        body=mapping
    )


# Encode with USE
def encode(text: str):
    embeddings = tf.make_ndarray(tf.make_tensor_proto(encoder([text]))).tolist()[0]
    return embeddings


def index_embedding_es(db: Elasticsearch(), data: pd.DataFrame()):
    for idx, row in tqdm(data.iterrows()):
        text = row[0]
        text_vector = encode(text)
        text_id = idx
        es_insert(db, text_id, text, text_vector)


# TEST
test = get_data(DATA_PATH)
db = es_connect(DB_HOST, DB_PORT)
index_embedding_es(db, test)


# WORKING!
# Right now it is storing vector for whole document as a sentence, Later use SIF to get vector for document
# Or, use Doc2Vec for getting 512-dim vectors for the documents


# SEARCH
def search_docsim(db: Elasticsearch(), text_vector: list()):
    search_results = []
    query = {"query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'document_vector')+1.0",
                "params": {"query_vector": text_vector}
            }
        }
    }}
    response = db.search(index='documents', body=query)

    for hit in response['hits']['hits']:
        search_result = {
            'similarity': hit['_score'],
            'document': hit['_source']['document']
        }
        search_results.append(search_result)
    return search_results


def normalize_score(scores):
    return [score / np.max(scores) for score in scores]


def search_db(db: Elasticsearch(), text, text_vector):
    search_results = []
    search_results = search_docsim(db, text_vector)

    # Convert score to percentage match
    search_results = [{'percentage_match': round((search_result['similarity'] / 2) * 100),
                       'document': search_result['document']
                       } for search_result in search_results]

    return search_results


def search(text, reload_data=False):
    if reload_data:
        data= get_data(DATA_PATH)
        es_setup(DB_HOST, DB_PORT)
        db = es_connect(DB_HOST, DB_PORT)
        index_embedding_es(db, data)
    else:
        db = es_connect(DB_HOST, DB_PORT)
    results = search_db(db, text, encode(text))
    return results


results = search(open("app/data/computers_1.txt", "r", encoding='UTF-8').read())
