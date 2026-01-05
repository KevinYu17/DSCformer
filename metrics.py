import torch
from monai.metrics import get_confusion_matrix

def compute_dice(y_pred: torch.Tensor, y: torch.tensor):
    y_pred = y_pred[:, 1].unsqueeze(1)
    y = y[:, 1].unsqueeze(1)
    return (2.0 * torch.sum(y * y_pred)) / (torch.sum(y) + torch.sum(y_pred))

def compute_f1_rec_pre_acc(y_pred: torch.Tensor, y: torch.tensor):
    confusion_matrix = get_confusion_matrix(y_pred, y, include_background=False)
    tp = confusion_matrix[0, 0, 0]
    fp = confusion_matrix[0, 0, 1]
    tn = confusion_matrix[0, 0, 2]
    fn = confusion_matrix[0, 0, 3]
    accuracy = (tp+tn)/(tp+fp+tn+fn+1e-8)
    precision = tp / (tp+fp+1e-8)
    recall = tp / (tp+fn+1e-8)
    f1 = 2*precision*recall / (precision+recall+1e-8)

    return f1, recall, precision, accuracy