# -*- coding: utf-8 -*-

import jieba
from jieba.analyse import *
import os,re
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools

def stopwordslist(filepath):    # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r').readlines()]    #以行的形式读取停用词表，同时转换为列表
    return stopword

filepath = '../baidu_stopwords.txt'
stopwordslist(filepath)  #调用函数

filepath = '../Manning.txt'
with open(filepath) as file_pi:
    contents = file_pi.read()

content1 = contents.replace(' ','')     # 去掉文本中的空格
            
pattern = re.compile("[^\u4e00-\u9fa5]")    #只保留中文
content2= re.sub(pattern,'',content1)      #把文本中匹配到的字符替换成空字符

jieba.suggest_freq(('進口貨'),tune=True)
jieba.suggest_freq(('水貨'),tune=True)
cutwords = jieba.lcut(content2)    #精确模式分词

stopwords = stopwordslist(filepath)     # 这里加载停用词的路径
words = ''
for word in cutwords:     #for循环遍历分词后的每个词语
    if word not in stopwords:     #判断分词后的词语是否在停用词表内
        if word != '\t':
            words += word
            words += "/"

n_gram_range = (1, 1)
# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords).fit([words])
candidates = count.get_feature_names_out()

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
doc_embedding = model.encode([words])
candidate_embeddings = model.encode(candidates)

top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords_1 = [candidates[index] for index in distances.argsort()[0][-top_n:]]


def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


keywords_2 = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=10, nr_candidates=10)


def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


keywords_3 = mmr(doc_embedding, candidate_embeddings, candidates, top_n=10, diversity=0.3)

d2 = {'Output': keywords_3}
print(d2)