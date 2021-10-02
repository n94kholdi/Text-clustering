import codecs
from collections import Counter
wordDict = Counter()
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
import operator
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import itertools
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import Training

class Test():

    def __init__(self, project_num, mode):
        super().__init__()
        self.Tags_test = []
        self.news_test = []
        self.news_str_test = []
        self.docs_represent = []
        self.unique_Tags = []
        self.predicts = []
        self.project_num = project_num
        self.mode = mode

    def Read_data(self):

        with codecs.open('Hamshahri/test.txt', 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('@@@@@@@@@@')
                self.Tags_test.append(text[0])
                self.news_test.append(text[1].strip().split())
                self.news_str_test.append(text[1])

        self.unique_Tags = list(set(self.Tags_test))

        Tags_dict = {self.unique_Tags[0]: 0, self.unique_Tags[1]: 1, self.unique_Tags[2]: 2, self.unique_Tags[3]: 3, self.unique_Tags[4]: 4}

        return self.Tags_test, self.news_test

    def represent_test_data(self):

        train = Training.Train(self.project_num, self.mode)
        train.news = self.news_test
        train.Tags = self.Tags_test
        train.unique_Tags = self.unique_Tags
        train.news_str = self.news_str_test

        if self.project_num == 'p1_part1':

            ### build vectors with Word2Vec :
            # train.word2vec()

            ### build doc with mean of words :
            train.build_doc_with_mean()

        if self.project_num == 'p1_part2':

            ### build vectors with Word2Vec :
            # train.word2vec()

            ### build doc with mean of words :
            train.build_doc_with_tf_idf()

        if self.project_num == 'p1_part3':

            ### build vectors with Word2Vec :
            # train.doc2vec()

            ### build doc with mean of words :
            train.build_doc_with_doc2vec()

        if self.project_num == 'p1_part4':
            ### compute svd for dimensionaly reduction :
            train.compute_svd()

        self.docs_represent = train.docs_represent

    def predict_test_text(self):

        cluster_centers = np.load(self.project_num + '\\' + 'train' + '\cluster_centers.npy')
        cluster_names = np.load(self.project_num + '\\' + 'train' + '\cluster_names.npy')

        Tags_dict = {self.unique_Tags[0]: 0, self.unique_Tags[1]: 1, self.unique_Tags[2]: 2, self.unique_Tags[3]: 3, self.unique_Tags[4]: 4}
        for i in range(0, len(self.Tags_test)):
            self.Tags_test[i] = Tags_dict[self.Tags_test[i]]

        for i in range(0, len(self.docs_represent)):

            Max = 0
            ind = 0
            for j in range(0, len(cluster_names)):
                # dist = np.linalg.norm(self.docs_represent[i] - cluster_centers[j])
                dist = abs(np.sum(self.docs_represent[i] * cluster_centers[j]))
                if Max < dist:
                    Max = dist
                    ind = j
            self.predicts.append(cluster_names[ind])

        a = 0
