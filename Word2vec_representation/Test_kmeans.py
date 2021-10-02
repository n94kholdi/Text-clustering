import Training
import Evaluation
import Testing
from sklearn.metrics import precision_recall_fscore_support

test = Testing.Test('p1_part1', 'test')
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
