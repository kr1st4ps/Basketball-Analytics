"""
Player utils.
"""

import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import KMeans


class Player:
    """
    Player class, containing all information of a detected and tracked player.
    """

    _id_counter = 0

    def __init__(self, start_frame, bbox, poly, team):
        type(self)._id_counter += 1
        self.id = self._id_counter

        self.bbox_history = [bbox]
        self.poly_history = [poly]

        self.team = team

        self.start_frame = start_frame
        self.last_seen = start_frame
        self.end_frame = None

    def __str__(self):
        return f"{self.id}.\tFrames[{self.start_frame}-{self.end_frame}]\tHistory{self.poly_history}"

    def __del__(self):
        None

    def __eq__(self, other):
        return self.id == other.id

    def update(self, bbox, poly, frame_id):
        """
        Updates player info.
        """
        self.bbox_history.insert(0, bbox)
        self.poly_history.insert(0, poly)
        self.last_seen = frame_id
        if len(self.bbox_history) > 5:
            self.bbox_history.pop()
            self.poly_history.pop()

    def lost(self, frame_id):
        """
        Handles player tracking being lost.
        """
        self.end_frame = frame_id
        #   TODO save to file


def bb_intersection_over_union(bbox_a, bbox_b):
    """
    Finds IoU (intersection over union) value of two bboxes.
    """
    # x_max = max(bbox_a[0], bbox_b[0])
    # y_max = max(bbox_a[1], bbox_b[1])
    # x_min = min(bbox_a[2], bbox_b[2])
    # y_min = min(bbox_a[3], bbox_b[3])

    # intersection_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)
    intersection_width = max(
        0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]) + 1
    )
    intersection_height = max(
        0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]) + 1
    )
    intersection_area = intersection_width * intersection_height

    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)

    return iou


def poly_intersection_over_union(poly_a, poly_b):
    poly_a = Polygon(poly_a)
    poly_b = Polygon(poly_b)
    i = poly_a.intersection(poly_b).area
    u = poly_a.union(poly_b).area
    iou = i / u

    return iou


def filter_bboxes(bboxes, polys, confidences, iou_threshold=0.5):
    """
    Filters out bounding boxes that overlap significantly with others.
    """
    filtered_bboxes = []
    filtered_polys = []
    filtered_confidences = []

    keep = [True] * len(bboxes)

    for i in range(len(bboxes)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue

            bbox_i_inside_j = all(
                bboxes[i][k] >= bboxes[j][k] for k in range(2)
            ) and all(bboxes[i][k] <= bboxes[j][k + 2] for k in range(2))
            bbox_j_inside_i = all(
                bboxes[j][k] >= bboxes[i][k] for k in range(2)
            ) and all(bboxes[j][k] <= bboxes[i][k + 2] for k in range(2))

            iou = bb_intersection_over_union(bboxes[i], bboxes[j])

            # if iou > iou_threshold or bbox_i_inside_j or bbox_j_inside_i:
            if iou > iou_threshold:
                # print(f"IoU: {iou}")
                if confidences[i] >= confidences[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    for i in range(len(bboxes)):
        if keep[i]:
            filtered_bboxes.append(bboxes[i])
            filtered_polys.append(polys[i])
            filtered_confidences.append(confidences[i])

    return filtered_bboxes, filtered_polys, filtered_confidences


def get_team(bbox, poly, img):
    c = (0, 0, 0)
    poly_mask = np.zeros_like(img[:, :, 0])
    poly = np.array([poly], dtype=np.int32)
    cv2.fillPoly(poly_mask, poly, color=255)

    bbox_mask = np.zeros_like(img[:, :, 0])
    cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, thickness=-1)

    combined_mask = cv2.bitwise_and(poly_mask, bbox_mask)

    nonzero_coords = np.column_stack(np.where(combined_mask != 0))

    for coord in nonzero_coords:
        x, y = coord
        c += img[x][y]

    return (c / len(nonzero_coords)).round()


def get_label(kmeans, value):
    color_lab = np.array(value).reshape(1, 1, 3).astype(np.uint8)
    color_lab = cv2.cvtColor(color_lab, cv2.COLOR_RGB2Lab)
    label = kmeans.predict(color_lab.reshape(-1, 3))[0]

    return label


def get_team_coef(bboxes, polys, img):
    all_colors = []
    for bbox, poly in zip(bboxes, polys):
        c = (0, 0, 0)
        bbox = [round(coord) for coord in bbox.numpy().tolist()]

        poly_mask = np.zeros_like(img[:, :, 0])
        poly = np.array([poly], dtype=np.int32)
        cv2.fillPoly(poly_mask, poly, color=255)

        bbox_mask = np.zeros_like(img[:, :, 0])
        cv2.rectangle(
            bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, thickness=-1
        )

        combined_mask = cv2.bitwise_and(poly_mask, bbox_mask)

        nonzero_coords = np.column_stack(np.where(combined_mask != 0))

        for coord in nonzero_coords:
            x, y = coord
            c += img[x][y]

        all_colors.append((c / len(nonzero_coords)).round())

    colors = np.array(all_colors)
    colors_lab = np.apply_along_axis(
        lambda x: np.reshape(np.array([x]), (1, 1, 3)).astype(np.uint8), 1, colors
    )
    colors_lab = np.array(all_colors).reshape(-1, 1, 3).astype(np.uint8)
    colors_lab = cv2.cvtColor(colors_lab, cv2.COLOR_RGB2Lab)

    num_clusters = 3
    X = colors_lab.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    # labels = kmeans.predict(X)

    # centroids = kmeans.cluster_centers_

    # # Create a new figure and 3D subplot
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")

    # x = np.array(labels == 0)
    # y = np.array(labels == 1)
    # z = np.array(labels == 2)

    # # Scatter plot for centroids and clusters
    # ax.scatter(
    #     centroids[:, 0],
    #     centroids[:, 1],
    #     centroids[:, 2],
    #     c="black",
    #     s=150,
    #     label="Centers",
    #     alpha=1,
    # )
    # ax.scatter(X[x, 0], X[x, 1], X[x, 2], c="blue", s=40, label="C1")
    # ax.scatter(X[y, 0], X[y, 1], X[y, 2], c="yellow", s=40, label="C2")
    # ax.scatter(X[z, 0], X[z, 1], X[z, 2], c="red", s=40, label="C3")
    # for i, (x_val, y_val, z_val) in enumerate(X):
    #     ax.text(
    #         x_val,
    #         y_val,
    #         z_val,
    #         f"{x_val:.2f}, {y_val:.2f}, {z_val:.2f}",
    #         color="black",
    #         fontsize=8,
    #     )

    # # Set labels and legend
    # ax.set_xlabel("L")
    # ax.set_ylabel("a")
    # ax.set_zlabel("b")
    # ax.set_title("K-means Clustering")
    # ax.legend()

    # # Save the plot as an image
    # plt.savefig("kmeans_clusters.png")
    # print("Clustered Data with Labels:")
    # for point, label in zip(X, labels):
    #     print(f"Data: {point}, Label: {label}")
    # sys.exit()

    return kmeans
