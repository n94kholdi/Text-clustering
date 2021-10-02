import Training
import Evaluation
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

train = Training.Train('p1_part3', 'train')
Tags, news = train.Read_data()

### build vectors with Word2Vec :
# print(1)
# train.doc2vec()

### build doc with mean of words :
print(2)
train.build_doc_with_doc2vec()

### predict clusters of train_set with kmeans :
print(3)
train.kmeans_predict()

### Evaluate model with predict_labels of kmeans :
print(4)
eval = Evaluation.Evaluate(train.Tags, train.kmeans_labels)

### (accuracy) & (NMI) & (V_measure) :
eval.acc_NMI_V_measure()

### (precision) & (recall) & (V_measure) :
##########################################
print('Micro_average evaluation :\n')

Precision = eval.precision('micro')
Recall = eval.recall('micro')

print('precision : ', Precision)
print('recall : ', Recall)
print('F1_measure : ', eval.F1_measure(Precision, Recall))
print()

#######################################
print('Macro_average evaluation :\n')

Precision = eval.precision('macro')
Recall = eval.recall('macro')

print('precision : ', Precision)
print('recall : ', Recall)
print('F1_measure : ', eval.F1_measure(Precision, Recall))
print()
