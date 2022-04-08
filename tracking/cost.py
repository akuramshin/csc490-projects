import numpy as np
from shapely.geometry import Polygon


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            x, y, l, w, yaw = bboxes1[i]
            alpha = np.pi/2 - yaw
            dx = np.array([(l/2)*np.cos(yaw), (w/2)*np.cos(alpha)])
            dy = np.array([(l/2)*np.sin(yaw), (w/2)*np.sin(alpha)])
            points = [(x+([1,-1]*dx).sum(), y+dy.sum()), (x+dx.sum(), y+([1,-1]*dy).sum()), (x+([-1,1]*dx).sum(), y-dy.sum()), (x-dx.sum(), y+([-1,1]*dy).sum())]
            b_1_i = Polygon(points)

            x, y, l, w, yaw = bboxes2[j]
            alpha = np.pi/2 - yaw
            dx = np.array([(l/2)*np.cos(yaw), (w/2)*np.cos(alpha)])
            dy = np.array([(l/2)*np.sin(yaw), (w/2)*np.sin(alpha)])
            points = [(x+([1,-1]*dx).sum(), y+dy.sum()), (x+dx.sum(), y+([1,-1]*dy).sum()), (x+([-1,1]*dx).sum(), y-dy.sum()), (x-dx.sum(), y+([-1,1]*dy).sum())]

            b_2_j = Polygon(points)

            if b_1_i.union(b_2_j).area != 0:
                iou = b_1_i.intersection(b_2_j).area / b_1_i.union(b_2_j).area
            else:
                iou = 0
            
            iou_mat[i, j] = iou


    return iou_mat
