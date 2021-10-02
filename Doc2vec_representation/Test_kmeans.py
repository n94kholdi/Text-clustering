import Evaluation
import Testing
from sklearn.metrics import precision_recall_fscore_support

test = Testing.Test('p1_part3', 'test')
Tags, news = test.Read_data()

## build test set with word2vec_mean :
test.represent_test_data()

## predict test set labels :
test.predict_test_text()

### Evaluate model with predict_labels of kmeans :
eval = Evaluation.Evaluate(test.Tags_test, test.predicts)

### (accuracy) & (NMI) & (V_measure) :
eval.acc_NMI_V_measure()

print('micro : ', precision_recall_fscore_support(test.Tags_test, test.predicts, average='micro'))
print('macro : ', precision_recall_fscore_support(test.Tags_test, test.predicts, average='macro'))