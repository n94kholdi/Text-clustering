import Training
import Evaluation
from sklearn.metrics import precision_recall_fscore_support

train = Training.Train('p1_part4', 'train')
Tags, news = train.Read_data()

### build doc with mean of words :
# print('build doc_word Mat :')

# train.build_docword_Mat_with_tf()
#
# doc_word = train.doc_word
# print(doc_word.shape)


### compute svd for dimensionaly reduction :
print('compute svd')
train.compute_svd()

### predict clusters of train_set with kmeans :
print('Run kmeans')
train.kmeans_predict()

### Evaluate model with predict_labels of kmeans :
eval = Evaluation.Evaluate(train.Tags, train.kmeans_labels)

### (accuracy) & (NMI) & (V_measure) :
eval.acc_NMI_V_measure()

print('micro : ', precision_recall_fscore_support(train.Tags, train.kmeans_labels, average='micro'))
print('macro : ', precision_recall_fscore_support(train.Tags, train.kmeans_labels, average='macro'))