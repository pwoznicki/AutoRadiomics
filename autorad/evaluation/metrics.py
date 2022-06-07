import monai
import numpy as np
import torch
from medpy import metric


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape
    )


class ConfusionMatrix:
    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError(
                "'test' and 'reference' must both be set to compute confusion matrix."
            )

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        ):
            if case is None:
                self.compute()
                break

        return (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        )


# Borrowed from https://github.com/MIC-DKFZ/nnUNet/blob/HEAD/nnunet/evaluation/metrics.py#L141-L175
def specificity(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    **kwargs
):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tn / (tn + fp))


def sensitivity(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    **kwargs
):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fn))


def dice(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    **kwargs
):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(2.0 * tp / (2 * tp + fp + fn))


def jaccard(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    **kwargs
):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fp + fn))


# Borrowed from https://github.com/MIC-DKFZ/nnUNet/blob/HEAD/nnunet/evaluation/metrics.py#L141-L175
def precision(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    **kwargs
):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fp))


# Borrowed from https://github.com/MIC-DKFZ/nnUNet/blob/HEAD/nnunet/evaluation/metrics.py#L141-L175
def hausdorff_distance_95(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


# Borrowed from https://github.com/MIC-DKFZ/nnUNet/blob/HEAD/nnunet/evaluation/metrics.py#L141-L175
def avg_surface_distance(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def calculate_monai_metrics(metric_names: list, y_true, y_pred):
    y_true_tensor = torch.unsqueeze(
        torch.unsqueeze(torch.tensor(y_true), 0), 0
    )
    y_pred_tensor = torch.unsqueeze(
        torch.unsqueeze(torch.tensor(y_pred), 0), 0
    )
    # include_background - whether to skip first channel
    confusion_matrix = monai.metrics.get_confusion_matrix(
        y_pred_tensor, y_true_tensor, include_background=False
    )
    result_metrics = []
    for metric_name in metric_names:
        result_tensor = monai.metrics.compute_confusion_matrix_metric(
            metric_name, confusion_matrix
        )
        result_metrics.append(result_tensor.item())

    return result_metrics
