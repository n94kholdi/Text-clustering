import Training
import Evaluation
from sklearn.metrics import precision_recall_fscore_support

train = Training.Train('p1_part1', 'train')
Tags, news = train.Read_data()

### build vectors with Word2Vec :
# train.word2vec()

### build doc with mean of words :
train.build_doc_with_mean()

### predict clusters of train_set with kmeans :
train.kmeans_predict()

### Evaluate model with predict_labels of kmeans :
eval = Evaluation.Evaluate(train.Tags, train.kmeans_labels)

### (accuracy) & (NMI) & (V_measure) :
eval.acc_NMI_V_measure()

print('micro : ', precision_recall_fscore_support(train.Tags, train.kmeans_labels, average='micro'))
print('macro : ', precision_recall_fscore_support(train.Tags, train.kmeans_labels, average='macro'))

#
# ### (precision) & (recall) & (V_measure) :
# ##########################################
# print('Micro_average evaluation :\n')
#
# Precision = eval.precision('micro')
# Recall = eval.recall('micro')
#
# print('precision : ', Precision)
# print('recall : ', Recall)
# print('F1_measure : ', eval.F1_measure(Precision, Recall))
# print()
#
# #######################################
# print('Macro_average evaluation :\n')
#
# Precision = eval.precision('macro')
# Recall = eval.recall('macro')
#
# print('precision : ', Precision)
# print('recall : ', Recall)
# print('F1_measure : ', eval.F1_measure(Precision, Recall))
# print()
