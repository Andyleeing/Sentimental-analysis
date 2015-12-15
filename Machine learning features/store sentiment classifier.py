#! /usr/bin/env python2.7
#coding=utf-8

"""
Use positive and negative review set as corpus to train a sentiment classifier.
This module use labeled positive and negative reviews as training set, then use nltk scikit-learn api to do classification task.
Aim to train a classifier automatically identifiy review's positive or negative sentiment, and use the probability as review helpfulness feature.

"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import MySQLdb
import textprocessing as tp
import pickle
import os
import itertools
from random import shuffle
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support



# file_list=os.listdir("/home/hadoop/iy")
# i=0
# max=0
# tname=''
# for name in file_list:
#     uipath = unicode("/home/hadoop/iy/"+name , "utf8")
#     txt_file1 = open(uipath, 'r')
#     txt_tmp1 = len(txt_file1.readlines())
#     if(txt_tmp1>=1000):
#         print name,txt_tmp1
#     if(txt_tmp1>max):
#         max=txt_tmp1
# #       tname=name
#
# print tname,max

# sig_re='太逗了 无语 帅帅的 我靠,人死了还有呼吸'
#
# seg_list = tp.segmentation(sig_re, 'list')


# 1. Load positive and negative review data
# pos_review = tp.seg_fil_txt("/home/hadoop/goodnew.txt")
# neg_review = tp.seg_fil_txt("/home/hadoop/badnew.txt")

#sentiment_review = tp.seg_fil_txt("/home/hadoop/iyTop10/万物生长")

sentiment_review = tp.get_txt_data("/home/hadoop/成人记2电影版客观性.txt",'lines')

posdict = tp.get_txt_data(
    "/home/hadoop/coding/Sentiment features/Sentiment dictionary features/sentiment dictionary/positive and negative dictionary/posdict.txt",
    "lines")
negdict = tp.get_txt_data(
    "/home/hadoop/coding/Sentiment features/Sentiment dictionary features/sentiment dictionary/positive and negative dictionary/negdict.txt",
    "lines")
stopwords = tp.get_txt_data('/home/hadoop/coding/stopword.txt', 'lines')
posdict.extend(negdict)

i=0
sen_cur=[]
p_center = open("/home/hadoop/建国大业客观性.txt",'w+')
for sig_re in sentiment_review:
    #sig_re='挺棒'
    flag=False
    seg_list = tp.segmentation(sig_re, 'list')
    for w in seg_list:
        if w in posdict:
            sen_cur.append(sig_re)  #主观句
            flag=True
            break
    if(flag==False):
        seg_lists=Seg(str(sig_re))#分词
        for w in seg_lists:
            if w in posdict:
                i+=1
                sen_cur.append(sig_re)  #主观句
                print w,'\t',sig_re
                flag=True
                break

    #if(flag==False):   #客观句
    #    p_center.write(sig_re + '\n')
    #   i+=1
    #    print sig_re
print i
p_center.close()


sentiment=[]
stopwords =tp.get_txt_data('/home/hadoop/coding/stopword.txt', 'lines')
for sig in sen_cur:
    fil =[word for word in sig if word not in stopwords and word!=' ']
    sentiment.append(fil)


pos = pos_review
neg = neg_review



"""
# Cut positive review to make it the same number of nagtive review (optional)

shuffle(pos_review)
size = int(len(pos_review)/2 - 18)

pos = pos_review[:size]
neg = neg_review

"""






# 2. Feature extraction function
# 2.1 Use all words as features
# def bag_of_words(words):
#     return dict([(word, True) for word in words])
#
#
# # 2.2 Use bigrams as features (use chi square chose top 200 bigrams)
# def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=500):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     return bag_of_words(bigrams)
#
#
# # 2.3 Use words and bigrams as features (use chi square chose top 200 bigrams)
# def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=500):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     return bag_of_words(words + bigrams)


# 2.4 Use chi_sq to find most informative features of the review
# 2.4.1 First we should compute words or bigrams information score

def create_word_scores():
    posdata = tp.seg_fil_txt("/home/hadoop/goodnew.txt")
    negdata = tp.seg_fil_txt("/home/hadoop/badnew.txt")
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in negWords:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores


def create_bigram_scores():
    posdata = tp.seg_fil_txt("/home/hadoop/goodnew.txt")
    negdata = tp.seg_fil_txt("/home/hadoop/badnew.txt")
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))
    bigram_finderr = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finderr.nbest(BigramAssocMeasures.chi_sq, 350000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 350000)
    pos = posBigrams
    neg = negBigrams
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores


# Combine words and bigrams and compute words and bigrams information scores
def create_word_bigram_scores():
    posdata = tp.seg_fil_txt("/home/hadoop/goodnew.txt")
    negdata = tp.seg_fil_txt("/home/hadoop/badnew.txt")
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finderr = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finderr.nbest(BigramAssocMeasures.chi_sq,350000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq,350000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# Choose word_scores extaction methods
# word_scores = create_word_scores()
# word_scores = create_bigram_scores()
# word_scores = create_word_bigram_scores()


# 2.4.2 Second we should extact the most informative words or bigrams based on the information score
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# 2.4.3 Third we could use the most informative words and bigrams as machine learning features
# Use chi_sq to find most informative words of the review
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

# Use chi_sq to find most informative bigrams of the review
def best_word_features_bi(words):
    return dict([(word, True) for word in nltk.bigrams(words) if word in best_words])

# Use chi_sq to find most informative words and bigrams of the review
def best_word_features_com(words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3

def extract_features(dataset):
    feat = []
    for i in dataset:
        feat.append(best_word_features_com(i))
    return feat

# 3. Transform review to features by setting labels to words in review
def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg']
        negFeatures.append(negWords)
    return negFeatures


word_scores = create_word_bigram_scores()
# word_scores = create_word_scores()
# word_scores = create_bigram_scores()
# #
best_words = find_best_words(word_scores, 500000) # Set dimension and initiallize most informative words
# #
posFeatures = pos_features(best_word_features_com)
negFeatures = neg_features(best_word_features_com)

# posFeatures = pos_features(best_word_features_bi)
# negFeatures = neg_features(best_word_features_bi)

# posFeatures = pos_features(best_word_features)
# negFeatures = neg_features(best_word_features)

# 4. Train classifier and examing classify accuracy
# Make the feature set ramdon
shuffle(posFeatures)
shuffle(negFeatures)

#75% of features used as training set (in fact, it have a better way by using cross validation function)
size_pos = int(len(pos_review) * 0.75)
size_neg = int(len(neg_review) * 0.75)

train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
test_set = posFeatures[size_pos:] + negFeatures[size_neg:]

test, tag_test = zip(*test_set)

#store Classifier


def clf_score(classifier):
    classifier = SklearnClassifier(classifier)  #在nltk 中使用scikit-learn 的接口
    classifier.train(train_set)    #训练分类器
    predict = classifier.classify_many(test)     #对开发测试集的数据进行分类，给出预测的标签
    return precision_recall_fscore_support(tag_test,predict)
    #return accuracy_score(tag_test, predict)     #对比分类预测结果和人工标注的正确结果，给出分类器准确度

# print  clf_score(BernoulliNB())
# #print 'GaussianNB`s accuracy is %f' %clf_score(GaussianNB())
# #print 'MultinomiaNB`s accuracy is %f' %clf_score(MultinomialNB())
# print  clf_score(LogisticRegression())
# #print 'SVC`s accuracy is %f' %clf_score(SVC(gamma=0.001, C=100., kernel='linear'))
# print  clf_score(LinearSVC())
# #print 'NuSVC`s accuracy is %f' %clf_score(NuSVC())


# 5. After finding the best classifier, then check different dimension classification accuracy
# def score(classifier):
#     classifier = SklearnClassifier(classifier)
#     classifier.train(trainset)
#     pred = classifier.classify_many(test)
#     return accuracy_score(tag_test, pred)
#
# dimention = ['300000','350000','500000','600000'] #有结果可知道，此时，当算法采用SVM，维度为2250时，准确率最好。
#
# for d in dimention:
#     word_scores = create_word_bigram_scores()
#     best_words = find_best_words(word_scores, int(d))
#
#     posFeatures = pos_features(best_word_features_com)
#     negFeatures = neg_features(best_word_features_com)
#
#     # Make the feature set ramdon
#     shuffle(posFeatures)
#     shuffle(negFeatures)
#
#     # 75% of features used as training set (in fact, it have a better way by using cross validation function)
#     size_pos = int(len(pos_review) * 0.75)
#     size_neg = int(len(neg_review) * 0.75)
#
#     trainset = posFeatures[:size_pos] + negFeatures[:size_neg]
#     testset = posFeatures[size_pos:] + negFeatures[size_neg:]
#
#     test, tag_test = zip(*testset)
#
#     print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB())
#     print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB())
#     print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression())
#    # print 'SVC`s accuracy is %f' %score(SVC())
#     print 'LinearSVC`s accuracy is %f' %score(LinearSVC())
#    # print 'NuSVC`s accuracy is %f' %score(NuSVC())
#     print



# 6. Store the best classifier under best dimension
def store_classifier(clf, trainset, filepath):
    classifier = SklearnClassifier(clf)
    classifier.train(trainset)

    pred = classifier.prob_classify_many(extract_features(sentiment))
    p_file = open(filepath,'w+') #把结果写入文档
    # for i in pred:
    #     p_file.write(str(i.prob('pos'))+' '+str(i.prob('neg')))
    for (i,j) in zip(pred,sen_cur):
        p_file.write(str(i.prob('pos'))+'\t'+str(i.prob('neg'))+'\t'+j + '\n')
    p_file.close()

    # pred2 = []
    # for i in pred:
    #     pred2.append([i.prob('pos'), i.prob('neg')])
    # for r in sentiment_review[:2]:
    #     print r
    #     print "pos probability score: %f" %pred2[sentiment_review.index(r)][0]
    #     print "neg probability score: %f" %pred2[sentiment_review.index(r)][1]
    #     print

    # use pickle to store classifier

# def classifer(filepath,good,bad,center):
#     p_file = open(filepath,'r')
#     p_good = open(good,'w')
#     p_bad = open(bad,'w')
#     p_center = open(center,'w')
#     spliarr=[]
#     while True:
#         line = p_file.readline()
#         if not line:
#             break
#         spliarr=line.split(' ',3)
#         linenum=spliarr[0]
#         goodscore=float(spliarr[1])
#         badscore=float(spliarr[2])
#         comment=spliarr[3]
#
#         if(abs(goodscore-badscore)<=0.04):
#             p_center.write(str(linenum).decode("unicode-escape")+' '+str(goodscore).decode("unicode-escape") + ' ' + str(badscore).decode("unicode-escape")+' '+str(comment).decode("unicode-escape") + '\n')
#         elif(badscore-goodscore>0.04):
#             p_bad.write(str(linenum).decode("unicode-escape")+' '+str(goodscore).decode("unicode-escape") + ' ' + str(badscore).decode("unicode-escape")+' '+str(comment).decode("unicode-escape") + '\n')
#         else:
#             p_good.write(str(linenum).decode("unicode-escape")+' '+str(goodscore).decode("unicode-escape") + ' ' + str(badscore).decode("unicode-escape")+' '+str(comment).decode("unicode-escape") + '\n')
#
#     p_file.close()
#     p_good.close()
#     p_bad.close()
#     p_center.close()


store_classifier(LogisticRegression(), train_set, '/home/hadoop/建国大业主观性.txt')  #SVM 只能判断极性，没有极性值

#classifer('/home/hadoop/result/result.txt2','/home/hadoop/result/good.txt2','/home/hadoop/result/bad.txt2','/home/hadoop/result/center.txt2')
