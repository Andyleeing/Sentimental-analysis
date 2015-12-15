
#coding=utf-8

"""
Use a stored sentiment classifier to identifiy review positive and negative probability.
This module aim to extract review sentiment probability as review helpfulness features.

"""


import textprocessing as tp
import pickle
import itertools
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import sklearn


# 1. Load data
review = tp.get_excel_data("/home/hadoop/coding/Review set/HTC Z710t_review_2013.6.5.xlsx",1,12, "data")
sentiment_review = tp.seg_fil_senti_excel("/home/hadoop/coding/Review set/Meizu MX_review_2013.6.7.xlsx", 1, 12)



# 2. Feature extraction method
# Used for transform review to features, so it can calculate sentiment probability by classifier
def create_words_bigrams_scores():
    posdata = tp.seg_fil_senti_excel("/home/hadoop/coding/Sentiment features/Machine learning features/seniment review set/pos_review.xlsx",1,1)
    negdata = tp.seg_fil_senti_excel("/home/hadoop/coding/Sentiment features/Machine learning features/seniment review set/neg_review.xlsx", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)#把文本变成双词搭配的形式
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)#使用了卡方统计的方法，选择排名前1000的双词
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)

    pos = posWords + posBigrams #词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word]+=1
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd[word]+=1
        #word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()#积极词的数量
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量
    # for k in word_scores:
    #     print k,word_scores[k]
    return word_scores  #包括了每个词和这个词的信息量





def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]  #把词按信息量倒序排序,number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words

# Initiallize word's information score and extracting most informative words
word_scores = create_words_bigrams_scores()
best_words = find_best_words(word_scores, 1000) # Be aware of the dimentions
# for w in best_words:
#     print w,

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#Testing
# dictd=best_word_features(review)
# for key in dictd:
#     print key,dictd[key]

# 3. Function that making the reviews to a feature set
def extract_features(dataset):
    feat = []
    for i in dataset:
        feat.append(best_word_features(i))
    return feat

#Testing
# dd=extract_features(review)
# for ww in dd:
#     print

# 4. Load classifier
clf = pickle.load(open('/home/hadoop/coding/Sentiment features/Machine learning features/sentiment_classifier.pkl'))
print clf
# Testing single review

pred = clf.batch_prob_classify(extract_features(sentiment_review[:2])) # An object contian positive and negative probabiliy

pred2 = []
for i in pred:
    pred2.append([i.prob('pos'), i.prob('neg')])

for r in review[:2]:
    print r
    print "pos probability score: %f" %pred2[review.index(r)][0]
    print "neg probability score: %f" %pred2[review.index(r)][1]
    print

    
# 5. Store review sentiment probabilty socre as review helpfulness features
def store_sentiment_prob_feature(sentiment_dataset, storepath):
	pred = clf.batch_prob_classify(extract_features(sentiment_dataset))
	p_file = open(storepath, 'w')
	for i in pred:
	    p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
	p_file.close()