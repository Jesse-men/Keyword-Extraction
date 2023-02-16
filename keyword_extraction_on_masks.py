# -*- coding: utf-8 -*-
"""Keyword Extraction on masks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZWcYU1nKG0kJ2Hm-cdbF-ew_jpI6uS7O
"""

from google.colab import drive
drive.mount('/content/drive')

import jieba
from jieba.analyse import *
import os,re
def stopwordslist(filepath):    # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r').readlines()]    #以行的形式读取停用词表，同时转换为列表
    return stopword

filepath = "/content/drive/My Drive/datasets/baidu_stopwords.txt"            
stopwordslist(filepath)  #调用函数

#filepath = '/content/drive/My Drive/datasets/cosmetic_test4.txt'
filepath = '/content/drive/My Drive/datasets/Manning.txt'
with open(filepath) as file_pi:
    contents = file_pi.read()
print('\n【原文本：】'+'\n'+contents) 

content1 = contents.replace(' ','')     # 去掉文本中的空格
print('\n【去除空格后的文本：】'+'\n'+content1)
            
pattern = re.compile("[^\u4e00-\u9fa5]")    #只保留中文
content2= re.sub(pattern,'',content1)      #把文本中匹配到的字符替换成空字符
print('\n【去除符号后的文本：】'+'\n'+ content2)

jieba.suggest_freq(('進口貨'),tune=True)
jieba.suggest_freq(('水貨'),tune=True)
cutwords = jieba.lcut(content2)    #精确模式分词
print ('\n【精确模式分词后:】'+ '\n'+"/".join(cutwords))
stopwords = stopwordslist(filepath)     # 这里加载停用词的路径
words = ''
for word in cutwords:     #for循环遍历分词后的每个词语
    if word not in stopwords:     #判断分词后的词语是否在停用词表内
        if word != '\t':
            words += word
            words += "/"
print('\n【去除停用词后的分词：】'+ '\n'+ words +'\n' )

from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1, 1)

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords).fit([words])
candidates = count.get_feature_names()
candidates

!pip install sentence-transformers
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('distilbert-base-nli-mean-tokens')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
doc_embedding = model.encode([words])
candidate_embeddings = model.encode(candidates)

from sklearn.metrics.pairwise import cosine_similarity

top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
keywords

#萬寧 #萬寧五十周年 #50周年 #健與美全因你 #周年慶典 #yuu會員專享 #水漾透明質酸 #面霜 #面膜 #精華素

import numpy as np
import itertools

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
max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=10, nr_candidates=10)

#萬寧 #萬寧五十周年 #50周年 #健與美全因你 #周年慶典 #yuu會員專享 #水漾透明質酸 #面霜 #面膜 #精華素

import numpy as np

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
mmr(doc_embedding, candidate_embeddings, candidates, top_n=10, diversity=0.3)

#萬寧 #萬寧五十周年 #50周年 #健與美全因你 #周年慶典 #yuu會員專享 #水漾透明質酸 #面霜 #面膜 #精華素