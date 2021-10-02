from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import v_measure_score as v_measure
from sklearn.metrics import confusion_matrix
import numpy as np

class Evaluate():

    def __init__(self, labels, predicts):
        super().__init__()
        self.labels = labels
        self.predicts = predicts
        self.conf_mat = confusion_matrix(labels, predicts)

    def acc_NMI_V_measure(self):

        acc = 0
        for i in range(0, len(self.labels)):

            if self.predicts[i] == self.labels[i]:
                acc += 1

        print('Accuracy :', acc / len(self.labels))
        print('NMI :', NMI(self.labels, self.predicts))
        print('v_measure :', v_measure(self.labels, self.predicts))

    def compute_tp_fp(self, conf_mat):

        tp = []
        fp = []
        tn = []
        fn = []

        for i in range(conf_mat.shape[0]):

            tp.append(conf_mat[i][i])
            fp.append(np.sum(conf_mat[:, i]) - conf_mat[i, i])
            tn.append(np.sum(conf_mat[i, :]) - conf_mat[i, i])
            fn.append(np.sum(np.sum(conf_mat)) - tp[i] - fp[i] - tn[i])

        return tp, fp, tn, fn

    def precision(self, type):

        tp, fp, tn, fn = self.compute_tp_fp(self.conf_mat)

        if type == 'micro':
            return np.sum(tp)/(np.sum(tp) + np.sum(fp))
        elif type == 'macro':

            each_precision = []
            for i in range(self.conf_mat.shape[0]):
                # a = (tp[i] + fp[i])
                # if a == 0:
                #     print('worst class :', i)
                # each_precision.append(tp[i]/(tp[i] + fp[i]))
                try:
                    each_precision.append(tp[i]/(tp[i] + fn[i]))
                except:
                    each_precision.append(0)
            return np.mean(each_precision)

    def recall(self, type):

        tp, fp, tn, fn = self.compute_tp_fp(self.conf_mat)

        if type == 'micro':
            return np.sum(tp)/(np.sum(tp) + np.sum(fn))

        elif type == 'macro':

            each_recall = []
            for i in range(self.conf_mat.shape[0]):
                each_recall.append(tp[i] / (tp[i] + fn[i]))

            return np.mean(each_recall)

    def F1_measure(self, recall, precision):

        return 2*precision*recall/(precision + recall)
