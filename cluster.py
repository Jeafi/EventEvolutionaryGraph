# -*- coding: utf-8 -*
import json
from gensim import corpora, models, similarities
from jieba.analyse import textrank
import jieba
import pickle
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm


def fetch_data():
    with open(r'processresult/capus.txt', 'r', encoding='utf8') as f:
        corpus = [json.loads(x) for x in f.readlines()]
        return corpus


def get_tfidf_and_lsi(corpus):
    _corpus = []
    for line in corpus:
        linetext = line['text_noun']
        linetext.extend(line['text_verb'])
        _corpus.append(linetext)
    corpus = _corpus
    dictionary = corpora.Dictionary(corpus)
    length_of_dictionary = len(dictionary)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # TF-IDF特征
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # LSI特征
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=500)
    lsi_vectors = lsi[tfidf_vectors]
    vec = []
    for i, ele in enumerate(lsi_vectors):
        feature = np.zeros(500)
        for idx, val in ele:
            feature[idx] = val
        vec.append(feature)
    return vec, lsi_vectors


def cluster(lsi):
    result = {}
    for i in tqdm(range(len(lsi))):
        if len(result) == 0:
            result[len(result)] = [i]
        else:
            feature_lsi_now = lsi[i]
            feature_lsi = []
            for key in result:
                ids = result[key]
                lsi_ = np.array([lsi[_id] for _id in ids])
                lsi_center = np.mean(lsi_, axis=0)
                feature_lsi.append(lsi_center)
            feature_lsi_now_t = torch.Tensor(feature_lsi_now).unsqueeze(0)
            feature_lsi_t = torch.Tensor(feature_lsi)
            feature_lsi_t = feature_lsi_t.view(-1, 500)
            sims_lsi = nn.functional.cosine_similarity(
                feature_lsi_t, feature_lsi_now_t)
            max_score, max_score_index = torch.max(sims_lsi, 0)
            max_score = max_score.item()
            max_score_index = max_score_index.item()
            if max_score >= 0.66:
                result[max_score_index].append(i)
            else:
                result[len(result)] = [i]
    return result


def write_file(result, corpus):
    fw = open(r'graph/demoGraph.json', 'w', encoding='utf8')
    for i in range(len(result)):
        re = result[i]
        data = dict()
        for r in re:
            for e in re:
                if corpus[r]['id'].replace('cause', '').replace('effect', '') == corpus[e]['id'].replace('effect', '').replace('cause', ''):
                    re.remove(r)
                    re.remove(e)
        if len(re) == 0:
            continue
        data['NodeNo'] = i
        data['eventCount'] = 0
        data['edge'] = []
        data['events'] = []
        data['eventCount'] = len(re)
        for r in re:
            data['events'].append(corpus[r])
            if "cause" in corpus[r]['id']:
                effect_serial = r + 1
                for _i in range(len(result)):
                    _re = result[_i]
                    for _r in _re:
                        if _r == effect_serial:
                            data['edge'].append(_i)
        json.dump(data, fw, ensure_ascii=False)
        fw.write('\n')


if __name__ == '__main__':
    corpus = fetch_data()
    lsi, _ = get_tfidf_and_lsi(corpus)
    result = cluster(lsi)
    write_file(result, corpus)
