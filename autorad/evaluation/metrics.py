from __future__ import annotations

import functools
from typing import Callable

import medpy.metric.binary as medpy_metrics

from autorad.utils import statistics


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape
    )


def get_medpy_metric(metric_name: str) -> Callable:
    if metric_name == "dice":
        return medpy_metrics.dc
    elif metric_name == "jaccard":
        return medpy_metrics.jc
    elif metric_name == "sensitivity":
        return medpy_metrics.sensitivity
    elif metric_name == "specificity":
        return medpy_metrics.specificity
    elif metric_name == "precision":
        return medpy_metrics.precision
    elif metric_name == "recall":
        return medpy_metrics.recall
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def calculate_single_metric(metric_name: str, y_true, y_pred):
    fn = get_medpy_metric(metric_name)
    return fn(y_pred, y_true)


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
