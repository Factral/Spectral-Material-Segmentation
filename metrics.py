import torch
from torchmetrics import ConfusionMatrix
import numpy as np

class Metrics:  
    def __init__(self, pre_fix):
        self.confmat = ConfusionMatrix(task="MULTICLASS", num_classes=16, ignore_index=255)
        self.segment_evaluator = SegmentEvaluator(is_sparse=False, categories=None)        
        self.pre_fix = pre_fix

    def compute(self):
        confmatrix = self.confmat.compute()
        _ , _ , acc , _, _ = self.segment_evaluator(confmatrix, pre_fix=self.pre_fix, verbose=True)
        return confmatrix, np.array(acc).item()
    
    def update(self, pred, label):
        self.confmat.update(pred, label)
    
    def reset(self):
        self.confmat.reset()

    def to(self, device):
        self.confmat.to(device)

def nanmean(v, *args, inplace=True, **kwargs):
    """
    calculate the mean of v that contains nan values.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class SegmentEvaluator:
    """
    Calculate the Pixel Acc and Mean Acc, and other metrics based on segmentation masks.
    """
    def __init__(self, is_sparse=False, categories=None):
        self.is_sparse = is_sparse
        self.categories = categories

    def __call__(self, confmat, pre_fix="train", verbose=True):
        if self.is_sparse:
            true_confmat = confmat[:-1, :-1]
        else:
            true_confmat = confmat

        # calculate pixel accuracy
        with torch.no_grad():
            correct = torch.diag(true_confmat).sum()
            total = true_confmat.sum()

            # pixel acc and mean class accuracy
            accuracy = correct / total
            acc_per_cls = (torch.diag(true_confmat) / true_confmat.sum(axis=1))
            mean_acc = nanmean(acc_per_cls)

            # iou
            intersection = torch.diag(true_confmat)
            union = true_confmat.sum(0) + true_confmat.sum(1) - intersection
            iou_per_cls = intersection.float() / union.float()
            iou_per_cls[torch.isinf(iou_per_cls)] = float('nan')
            miou = nanmean(iou_per_cls)

        if verbose:
            print(pre_fix + "_acc", accuracy.tolist())
            print(pre_fix + "_mean_acc", mean_acc.tolist())
            print(pre_fix + "_miou", miou.tolist())

            # log per category performance
            if self.categories is not None:
                for idx, acc in enumerate(acc_per_cls.cpu().numpy()):
                    print(pre_fix + "_acc_cat_" + self.categories[idx], acc)

        return accuracy.tolist(), acc_per_cls.tolist(), mean_acc.tolist(), iou_per_cls.tolist(), miou.tolist()