import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

class Metrics:  
    def __init__(self, pre_fix):
        self.pre_fix = pre_fix
        self.confusion_matrix = np.zeros((46, 46))     
        self.ignore_index = 255

    def compute(self):
        pixel_acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-9)
        per_class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-9)
        macc = per_class_acc.sum() / ((np.sum(self.confusion_matrix, axis=1) > 0).sum() + 1e-9)
        per_class_iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
            + 1e-9
        )
        miou = per_class_iou.sum() / ((np.sum(self.confusion_matrix, axis=1) > 0).sum() + 1e-9)

        return pixel_acc, macc, miou
    
    def update(self, pred, label):

        gt = label.cpu().numpy()
        mask = gt != self.ignore_index

        pred = pred.cpu().numpy()

        self.confusion_matrix += sklearn_confusion_matrix(
            gt[mask], pred[mask], labels=[_ for _ in range(46)]
        )

    
    def reset(self):
        self.confusion_matrix = np.zeros((46, 46))     


