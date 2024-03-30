from itertools import combinations
import sys
import cv2
import numpy as np

from utils.constants import *
from utils.d2 import *

#   Dictionary for storing keypoint coordinates in the frame
frame_court_kp = {
    "TOP_LEFT": None, 
    "TOP_LEFT_HASH": None,
    "TOP_MID": None,
    "TOP_RIGHT_HASH": None,
    "TOP_RIGHT": None,

    "RIGHT_FT_TOP_RIGHT": None,
    "RIGHT_FT_TOP_LEFT": None,
    "RIGHT_FT_BOTTOM_LEFT": None,
    "RIGHT_FT_BOTTOM_RIGHT": None,

    "BOTTOM_RIGHT": None,
    "BOTTOM_RIGHT_HASH": None,
    "BOTTOM_MID": None,
    "BOTTOM_LEFT_HASH": None,
    "BOTTOM_LEFT": None,
    
    "LEFT_FT_BOTTOM_LEFT": None,
    "LEFT_FT_BOTTOM_RIGHT": None,
    "LEFT_FT_TOP_RIGHT": None,
    "LEFT_FT_TOP_LEFT": None,
    
    "CENTER_TOP": None,
    "CENTER_BOTTOM": None,
}
kp_keys = list(frame_court_kp.keys())

lk_params = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def get_roi_bbox_around_point(image_size, point, width, height):
    x, y = point
    half_width = width // 2
    half_height = height // 2
    x_min = max(0, x - half_width)
    y_min = max(0, y - half_height)
    x_max = min(image_size[1], x + half_width)  # Adjusted to use image height
    y_max = min(image_size[0], y + half_height)  # Adjusted to use image width
    return x_min, y_min, x_max, y_max

def find_point(point, H):
    point_orig = np.array(point)
    point_image2_homogeneous = np.dot(H, point_orig)
    point_image2_normalized = point_image2_homogeneous / point_image2_homogeneous[2]
    point_image2_euclidean = (point_image2_normalized[0], point_image2_normalized[1])

    return (int(point_image2_euclidean[0][0]), int(point_image2_euclidean[1][0]))

def area_of_polygon(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area

#   Finds largest polygon from user given points
def largest_polygon_area(coords_dict):
    valid_coords = [(key, coord) for key, coord in coords_dict.items() if coord is not None]

    max_area = 0
    largest_polygon = None
    order_of_keys = None

    for i in range(4, len(valid_coords) + 1):
        for combo in combinations(valid_coords, i):
            vertices = [coord for _, coord in combo]
            area = area_of_polygon(vertices)
            if area > max_area:
                max_area = area
                largest_polygon = combo
                order_of_keys = [key for key, _ in combo]

    if largest_polygon is None:
        return None, None, None
    else:
        return largest_polygon, max_area, order_of_keys
    

def is_point_inside_frame(point, frame_size):
    x, y = point
    width, height = frame_size
    return 0 <= x < width and 0 <= y < height

#   Saves coordinates of user selected point
def select_point(event, x, y, flags, params):
    global frame_court_kp, current_key
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_court_kp[current_key] = (x, y)

#   Opens video
cap = cv2.VideoCapture('test_video.mp4')

#   Initializes detectron2 model
detectron = myDetectron("small")

#   Allows user to find a frame on which to put keypoints
cv2.namedWindow('Frame')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(frame.shape)
    
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('n'):
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 24)
        continue

    elif key == ord('p'):
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
        continue
        
    elif key == ord('q'):
        break
result = frame.copy()
cv2.destroyAllWindows()

#   User inputs all visible keypoints using his mouse
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_point)
for current_key in kp_keys:
    while True:
        frame_copy = frame.copy()
        
        for key, point in frame_court_kp.items():
            if point is not None:
                cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                cv2.putText(frame_copy, key, (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame_copy, f"Place {current_key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Image', frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n'):
            break
        
        if key == ord('q'):
            print(frame_court_kp)
            cv2.destroyAllWindows()
            quit()

#   Finds largest polygon
largest_polygon, max_area, order_of_keys = largest_polygon_area(frame_court_kp)
if largest_polygon is None:
    print("Not enough points") #TODO try again
    sys.exit(0)

#   Collects respective points from the flat court image
flat_img = []
frame_img = []
for point in largest_polygon:
    frame_img.append([point[1][0], point[1][1]])
    flat_img.append(REAL_COURT_KP[point[0]])

#   Generates homography matrixes both ways
H_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))
H_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))

#   Finds coordinates of other points
keys_with_none_values = [key for key, value in frame_court_kp.items() if value is None]
keypoints_in_frame = {key: value for key, value in frame_court_kp.items() if value is not None}
for key in keys_with_none_values:
    frame_court_kp[key] = find_point([[REAL_COURT_KP[key][0]], [REAL_COURT_KP[key][1]], [1]], H_frame)

    #       FURTHER TESTING NEEDED - a point was placed above the digital scoreboard, ie it's not good for tracking
    #   Check if any new found points are inside the frame
    # if is_point_inside_frame(frame_court_kp[key], frame.shape[:2]):
    #     keypoints_in_frame[key] = frame_court_kp[key]

#   Find all persons on the court
outputs = detectron.get_shapes(frame, 0)
outputs = filter_predictions_by_court(outputs, [frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], frame_court_kp["BOTTOM_LEFT"]], frame.shape[:2])
annotated_img = detectron.get_annotated_image(frame, outputs)

#   Find good court keypoints (for next frame)
keypoints_to_track = {}
for key, point in keypoints_in_frame.items():
    point_roi_bbox = get_roi_bbox_around_point(frame.shape[:2], point, 30, 30)
    
    is_good_point = True
    for bbox in outputs["instances"].pred_boxes:
        if are_bboxes_intersect(point_roi_bbox, bbox):
            is_good_point = False
            break

    if is_good_point:
        keypoints_to_track[key] = point

#   TODO Try to detect court and players in next frame

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     #   Finds largest polygon
#     largest_polygon, max_area, order_of_keys = largest_polygon_area(frame_court_kp)
#     if largest_polygon is None:
#         print("Not enough points") #TODO try again
#         sys.exit(0)

#     #   Collects respective points from the flat court image
#     flat_img = []
#     frame_img = []
#     for point in largest_polygon:
#         frame_img.append([point[1][0], point[1][1]])
#         flat_img.append(REAL_COURT_KP[point[0]])

#     #   Generates homography matrixes both ways
#     H_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))
#     H_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))

#     #   Finds coordinates of other points
#     keys_with_none_values = [key for key, value in frame_court_kp.items() if value is None]
#     for key in keys_with_none_values:
#         frame_court_kp[key] = find_point([[REAL_COURT_KP[key][0] - frame_img[0][0]], [REAL_COURT_KP[key][1] - frame_img[0][1]], [1]], H_frame)

#   Warps frame to look flat
# im_dst = cv2.warpPerspective(result, H_flat, (2800, 1500))

#   Saves flattened image and original image
# cv2.imwrite("court_flat1.png", im_dst)
# cv2.imwrite('Result1.png', frame_copy)

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
