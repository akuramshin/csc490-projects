from dataclasses import dataclass
from nis import match
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.



    detections = torch.tensor([])
    labels = torch.tensor([])
    scores = torch.tensor([])
    for i, frame in enumerate(frames):
        batch_detections = frame.detections.centroids
        detections = torch.cat((detections, batch_detections), dim=0)
        batch_labels = frame.labels.centroids
        labels = torch.cat((labels, batch_labels), dim=0)
        batch_scores = frame.detections.scores
        scores = torch.cat((scores, batch_scores), dim=0)

    D = detections.shape[0]
    L = labels.shape[0]
    precision = torch.zeros(D)
    recall = torch.zeros(D)
    # from high to low
    _ , indices = torch.sort(scores, descending=True)
    sorted_detections = detections[indices, :]
    
    # whether distance between i-th detection and j-th label  <= threshold is in truth_table[j][i]
    # 
    # whether i-th detection matches j-th label                            is in match_table[j][i]
    #
    #                        detection
    #         [   [                              ],
    # labels      [                              ] ]
    #
    truth_table = ((sorted_detections.reshape(1, D, 2) - labels.reshape(L, 1, 2)).norm(dim=-1) <= threshold).long()

    # in each row, only first 1 should be kept. Remove others 
    max_ix_d = truth_table.argmax(dim=-1)
    max_ix_2d = torch.stack([torch.arange(L), max_ix_d], dim=-1)
    match_table = torch.zeros(truth_table.shape)
    match_table[max_ix_2d[:, 0], max_ix_2d[:, 1]] = 1
    indices = truth_table.sum(dim=-1) == 0
    match_table[indices] = 0
    
    for i in range(D):
        TP = match_table[:i+1, :].any(dim=0).sum().item()
        FP = i + 1 - TP
        FN = L - match_table[:i+1, :].any(dim=1).sum().item()

        precision[i] = TP / (TP + FP)
        recall[i] = TP / (TP + FN)

    return PRCurve(precision, recall)


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    precision = curve.precision
    recall = curve.recall

    recall_shifted = torch.cat((torch.tensor([0]), recall))

    return torch.sum(precision * (recall - recall_shifted[:-1])).item()


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(curve)

    return AveragePrecisionMetric(ap, curve)