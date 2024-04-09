"""
Court utils.
"""

import json
import os
import sys
import cv2
import numpy as np

from utils.functions import is_point_in_frame


def get_keypoints(kp_filename, frame, region="court"):
    """
    Loads in (if exists) or asks user to input keypoints
    """

    class KeypointData:
        """
        Helper class for storing keypoints.
        """

        def __init__(self):
            self.keypoints = {}
            self.current_keypoint = None

    def select_point(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if param.current_keypoint is not None:
                param.keypoints[param.current_keypoint] = (x, y)

    keypoint_data = KeypointData()

    if region == "court":
        keypoint_names = [
            "TOP_LEFT",
            "TOP_LEFT_HASH",
            "TOP_MID",
            "TOP_RIGHT_HASH",
            "TOP_RIGHT",
            "RIGHT_FT_TOP_RIGHT",
            "RIGHT_FT_TOP_LEFT",
            "RIGHT_FT_BOTTOM_LEFT",
            "RIGHT_FT_BOTTOM_RIGHT",
            "BOTTOM_RIGHT",
            "BOTTOM_RIGHT_HASH",
            "BOTTOM_MID",
            "BOTTOM_LEFT_HASH",
            "BOTTOM_LEFT",
            "LEFT_FT_BOTTOM_LEFT",
            "LEFT_FT_BOTTOM_RIGHT",
            "LEFT_FT_TOP_RIGHT",
            "LEFT_FT_TOP_LEFT",
            "CENTER_TOP",
            "CENTER_BOTTOM",
            "VB_TOP_LEFT",
            "VB_TOP_LEFT_MID",
            "VB_TOP_MID",
            "VB_TOP_RIGHT_MID",
            "VB_TOP_RIGHT",
            "VB_BOTTOM_RIGHT",
            "VB_BOTTOM_RIGHT_MID",
            "VB_BOTTOM_MID",
            "VB_BOTTOM_LEFT_MID",
            "VB_BOTTOM_LEFT",
        ]
    elif region == "scoreboard":
        keypoint_names = ["TL", "TR", "BR", "BL"]

    keypoint_data.keypoints = {name: None for name in keypoint_names}

    if os.path.exists(kp_filename):
        with open(kp_filename, "r", encoding="utf-8") as file:
            loaded_keypoints = json.load(file)
            keypoint_data.keypoints.update(loaded_keypoints)
    else:
        window_name = f"Placing {region} keypoints..."
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_point, keypoint_data)

        for keypoint in keypoint_names:
            keypoint_data.current_keypoint = keypoint
            while True:
                frame_copy = frame.copy()
                for name, point in keypoint_data.keypoints.items():
                    if point:
                        cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                        cv2.putText(
                            frame_copy,
                            name,
                            (point[0], point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                cv2.putText(
                    frame_copy,
                    f"Place {keypoint}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame_copy)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    sys.exit(0)

        with open(kp_filename, "w", encoding="utf-8") as file:
            json.dump(keypoint_data.keypoints, file)

        cv2.destroyAllWindows()

    return keypoint_data.keypoints


def get_court_poly(court_keypoints, frame_shape):
    """
    Gets court polygon.

    Note:
        This is being done, because when camera pans to one side of
        the court, furthest point away is being calculated as weird
        numbers, so we use the next closest point (which will not be
        in the frame anyway).
    """

    top_left = court_keypoints["TOP_LEFT"]
    top_right = court_keypoints["TOP_RIGHT"]
    if is_point_in_frame(top_left, frame_shape[1], frame_shape[0]):
        bottom_left = court_keypoints["BOTTOM_LEFT"]
    else:
        bottom_left = court_keypoints["BOTTOM_LEFT_HASH"]
    if is_point_in_frame(top_right, frame_shape[1], frame_shape[0]):
        bottom_right = court_keypoints["BOTTOM_RIGHT"]
    else:
        bottom_right = court_keypoints["BOTTOM_RIGHT_HASH"]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
    ).reshape((-1, 1, 2))


def draw_court_point(frame, point, key):
    """
    Draws a given point on the court.
    """

    cv2.putText(
        frame,
        key,
        (round(point[0]), round(point[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (round(point[0]), round(point[1])),
        5,
        (0, 255, 0),
        -1,
    )

    return frame
