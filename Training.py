import codecs
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
import operator
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import itertools
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
wordDict = Counter()


class Train():

    def __init__(self, project_num, mode):
        super().__init__()
        self.Tags = []
        self.news = []
        self.news_str = []
        self.docs_represent = []
        self.kmeans_labels = []
        self.unique_Tags = []
        self.doc_word = {}
        self.documents = []
        self.project_num = project_num

        self.mode = mode

    def Read_data(self):

        with codecs.open('Hamshahri/train.txt', 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('@@@@@@@@@@')
                self.Tags.append(text[0])
                self.news.append(text[1].strip().split())
                self.news_str.append(text[1])

        self.unique_Tags = list(set(self.Tags))

        return self.Tags, self.news

    def word2vec(self):

        model = Word2Vec(self.news, size=300, window=4, min_count=1, workers=4)
        model.train(self.news, total_examples=len(self.news), epochs=10)
        model.save(self.project_num + '\\' + self.mode + "\word2vec.model")

    def doc2vec(self):

        for i in range(0, len(self.news)):
            self.documents.append(TaggedDocument(self.news[i], [i]))

        model = Doc2Vec(self.documents, vector_size=300, window=4, min_count=1, workers=4)
        model.train(self.documents, total_examples=len(self.documents), epochs=30)
        model.save(self.project_num + '\\' + self.mode + "\doc2vec_model")

    def build_doc_with_doc2vec(self):

        model = Doc2Vec.load(self.project_num + '\\' + self.mode + "\doc2vec_model")

        for i in range(0, len(self.news)):
            self.docs_represent.append(model.docvecs[i])

    def build_doc_with_mean(self):

        model = Word2Vec.load(self.project_num + '\\' + self.mode + "\word2vec.model")

        for text in self.news:
            avg = np.zeros(len(model.wv[text[0]]))
            for word in text:
                vector = model.wv[word]  # numpy vector of a word
                avg += vector
            self.docs_represent.append(avg / len(text))


    def build_doc_with_tf_idf(self):

        model = Word2Vec.load(self.project_num + '\\' + self.mode + "\word2vec.model")

        self.doc_word = {self.unique_Tags[0]: {}, self.unique_Tags[1]: {}, self.unique_Tags[2]: {}, self.unique_Tags[3]: {}, self.unique_Tags[4]: {}}

        for i in range(0, len(self.news)):
            for j in range(0, len(self.news[i])):
                a = self.news[i][j]
                b = self.doc_word[self.Tags[i]].keys()
                c = 0
                if self.news[i][j] in self.doc_word[self.Tags[i]].keys():
                    self.doc_word[self.Tags[i]][self.news[i][j]] += 1
                else:
                    self.doc_word[self.Tags[i]][self.news[i][j]] = 1

        for i in range(0, len(self.news)):
            text = self.news[i]
            avg = np.zeros(len(model.wv[text[0]]))

            for word in text:
                vector = model.wv[word]  # numpy vector of a word
                tf = 1 + math.log(self.doc_word[self.Tags[i]][word], 10)
                df = 0
                for tag in self.doc_word.keys():
                    if word in self.doc_word[tag].keys():
                        df += self.doc_word[tag][word]
                idf = math.log((len(self.news)/df), 10)
                avg += tf * idf * vector

            self.docs_represent.append(avg / len(text))

    def build_docword_Mat_with_tf(self):

        # self.doc_word = {self.unique_Tags[0]: {}, self.unique_Tags[1]: {}, self.unique_Tags[2]: {}, self.unique_Tags[3]:{}, self.unique_Tags[4]: {}}
        # for i in range(0,len(self.news)):
        #     for j in range(0, len(self.news[i])):
        #         if self.news[i][j] in self.doc_word[self.Tags[i]].keys():
        #             self.doc_word[self.Tags[i]][self.news[i][j]] += 1
        #         else:
        #             self.doc_word[self.Tags[i]][self.news[i][j]] = 1
        #
        # for i in range(0, len(self.news)):
        #
        #     text = self.news[i]
        #     for word in text:
        #         tf = 1 + math.log(self.doc_word[self.Tags[i]][word], 10)
        #         self.doc_word[self.Tags[i]][word] = tf

        unique_words = list(set(list(itertools.chain.from_iterable(self.news))))
        unique_map = {}
        for i in range(0, len(unique_words)):
            unique_map[unique_words[i]] = i
        matrix = np.zeros((len(self.news), len(unique_words)))

        for i in range(0, len(self.news)):
            text = np.array(self.news[i])

            for word in text:
                ind = unique_map[word]
                matrix[i, ind] += 1

            for word in text:
                try:
                    ind = unique_map[word]
                    tf = 1 + math.log(matrix[i, ind], 10)
                    matrix[i, ind] = tf
                except:
                    pass

        self.doc_word = matrix

    def compute_svd(self):

        # sparse_doc_word = sparse.csc_matrix(self.doc_word)
        vectorizer = TfidfVectorizer(use_idf=False)
        sparse_doc_word = vectorizer.fit_transform(self.news_str)
        svd = TruncatedSVD(n_components=300)
        # svd.fit(sparse_doc_word)

        self.docs_represent = svd.fit_transform(sparse_doc_word)

    def kmeans_predict(self):

        kmeans = KMeans(n_clusters=len(self.unique_Tags), random_state=0).fit(self.docs_represent)
        self.kmeans_labels = list(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        cluster_names = [0, 1, 2, 3, 4]

        Tags_dict = {self.unique_Tags[0]: 0, self.unique_Tags[1]: 1, self.unique_Tags[2]: 2, self.unique_Tags[3]: 3, self.unique_Tags[4]: 4}
        count_tags_first = {self.unique_Tags[0]: 0, self.unique_Tags[1]: 0, self.unique_Tags[2]: 0, self.unique_Tags[3]: 0, self.unique_Tags[4]: 0}

        for i in range(0, len(self.unique_Tags)):

            count_tags = count_tags_first
            count_tags = {}
            for j in range(0, len(self.kmeans_labels)):
                if self.kmeans_labels[j] == i:
                    if self.Tags[j] in count_tags.keys():
                        count_tags[self.Tags[j]] += 1
                    else:
                        count_tags[self.Tags[j]] = 1

            best_label = max(count_tags.items(), key=operator.itemgetter(1))[0]
            count_tags_first[best_label] = -10000
            cluster_names[i] = Tags_dict[best_label]

            for j in range(0, len(self.kmeans_labels)):
                if self.kmeans_labels[j] == i:
                    self.kmeans_labels[j] = best_label

        for i in range(0, len(self.kmeans_labels)):
            self.kmeans_labels[i] = Tags_dict[self.kmeans_labels[i]]
            self.Tags[i] = Tags_dict[self.Tags[i]]

        np.save(self.project_num + '\\' + 'train' + '\cluster_centers.npy', cluster_centers)
        np.save(self.project_num + '\\' + 'train' + '\cluster_names.npy', cluster_names)
