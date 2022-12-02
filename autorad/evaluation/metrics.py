import functools

import monai
import torch

from autorad.utils import statistics


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape
    )


def calculate_single_metric(metric_name: str, y_true, y_pred):
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
    result_tensor = monai.metrics.compute_confusion_matrix_metric(
        metric_name, confusion_matrix
    )
    return result_tensor.item()


def calculate_metrics(metric_names: list, y_true, y_pred):
    result_metrics = []
    for metric_name in metric_names:
        result_metrics.append(
            calculate_single_metric(metric_name, y_true, y_pred)
        )

    return result_metrics


def calculate_metrics_bootstrap(metric_names: list, y_true, y_pred):
    result_metrics = []
    for metric_name in metric_names:
        stat, lower, upper = statistics.bootstrap_statistic(
            functools.partial(calculate_single_metric, metric_name),
            y_true,
            y_pred,
        )
        result_metrics.append((stat, lower, upper))
    return result_metrics
