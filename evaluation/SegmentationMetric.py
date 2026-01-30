from typing import Optional
import numpy as np

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

"""
# 横轴预测结果，纵轴真实情况
混淆矩阵      P           N
P           TP(真正例)    FN(假反例)
N           FP(假正例)    TN(真反例)
"""

class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix[:] = 0

    def _to_numpy(self, x):
        if _has_torch and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_confusion_matrix(self, label_true, label_pred):
        # label_true, label_pred are 1D numpy arrays
        mask = (label_true >= 0) & (label_true < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (label_true != self.ignore_index)
        label_true = label_true[mask].astype(int) # 真实标签
        label_pred = label_pred[mask].astype(int) # 预测标签
        if label_true.size == 0:
            return np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        combined = self.num_classes * label_true + label_pred # 将每种情况进行唯一编码
        counts = np.bincount(combined, minlength=self.num_classes * self.num_classes)
        return counts.reshape((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """Update confusion matrix with prediction and target.

        pred and target can be:
        - 2D arrays of shape (H, W) with class ids
        - 3D arrays with channel-first probabilities/logits: (C, H, W)
        - If batch of images, pass flattened/batched arrays yourself or call update per-sample.
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)

        # If pred is probabilities/logits with channel dim
        if pred.ndim == 3 and pred.shape[0] == self.num_classes:
            pred = np.argmax(pred, axis=0)

        # If target has channel dim (one-hot), reduce
        if target.ndim == 3 and target.shape[0] == self.num_classes:
            target = np.argmax(target, axis=0)

        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        label_pred = pred.flatten()
        label_true = target.flatten()

        self.confusion_matrix += self.get_confusion_matrix(label_true, label_pred)

    def pixel_accuracy(self):
        # 像素准确率 PA，即准确率（Accuracy）
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        cm = self.confusion_matrix
        correct = np.diag(cm).sum()
        total = cm.sum()
        if total == 0:
            return 0.0
        return float(correct) / float(total)

    def class_pixel_accuracy(self):
        # 类别像素准确率 CPA，即精准率（Precision）
        # Precision = TP / (TP + FP) 或 TN / (TN + FN)
        cm = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'): # 忽略除零错误和无效操作
            class_acc = np.diag(cm) / cm.sum(axis=1) # 沿axis=1的轴进行压缩
        class_acc = np.nan_to_num(class_acc) # 将NaN替换为0.0
        return class_acc # 返回一个一维数组，元素值表示每个类别的预测准确率

    def mean_pixel_accuracy(self):
        # 平均像素准确率 MPA
        class_acc = self.class_pixel_accuracy()
        valid = (self.confusion_matrix.sum(axis=1) > 0) # class_acc的mask
        if valid.sum() == 0:
            return 0.0
        return float(class_acc[valid].mean())

    def iou(self):
        # 交并比 IoU
        cm = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            intersection = np.diag(cm) # 取对角线元素值
            union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
            iou = intersection / union
        iou = np.nan_to_num(iou)
        return iou # 返回一个一维数组，元素值表示每个类别的IoU

    def mean_iou(self):
        # 平均交并比 MIoU
        iou = self.iou()
        valid = (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) > 0 # iou的mask
        if valid.sum() == 0:
            return 0.0
        return float(iou[valid].mean())

    def frequency_weighted_iou(self):
        # 频权交并比 FWIoU
        cm = self.confusion_matrix
        freq = cm.sum(axis=1) / cm.sum()
        iou = self.iou()
        return float((freq * iou).sum())

    def compute(self):
        """Return a dict with all metrics."""
        return {
            'PixelAccuracy': self.pixel_accuracy(),
            'ClassPixelAccuracy': self.class_pixel_accuracy(),
            'MeanPixelAccuracy': self.mean_pixel_accuracy(),
            'IoU': self.iou(),
            'MeanIoU': self.mean_iou(),
            'FWIoU': self.frequency_weighted_iou(),
        }

if __name__ == '__main__':
    # Quick self-check on synthetic data
    metric = SegmentationMetric(num_classes=3, ignore_index=None)
    # ground truth: class 0 region, class1 region, class2 region
    gt = np.array([
        [0,0,1,1],
        [0,0,1,2],
        [2,2,2,2],
        [0,1,1,2]
    ], dtype=np.int64)
    # prediction with some errors
    pred = np.array([
        [0,1,1,1],
        [0,0,1,2],
        [2,2,1,2],
        [0,1,0,2]
    ], dtype=np.int64)
    metric.update(pred, gt)
    res = metric.compute()
    print('Confusion matrix:\n', metric.confusion_matrix)
    for k,v in res.items():
        print(k, v)