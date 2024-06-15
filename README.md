# Basketball-Analytics
Software that tracks basketball players on the court from standard broadcast view videos. Made for my bachelor thesis (Riga Techincal University, 2024).

# Result example
- Modified original video:
![Modified original video](https://raw.githubusercontent.com/kr1st4ps/Basketball-Analytics/main/resources/runs/output_test_video.mp4)
<video width="320" height="240" controls>
  <source src="https://raw.githubusercontent.com/kr1st4ps/Basketball-Analytics/main/resources/runs/output_test_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
- Genarated top view video:
[![Watch the video](https://raw.githubusercontent.com/kr1st4ps/Basketball-Analytics/main/resources/runs/output_test_video.mp4)](https://raw.githubusercontent.com/kr1st4ps/Basketball-Analytics/main/resources/runs/output_test_video.mp4)
- Collected information JSON:
```json
...{
  "id": [
      11,
      15,
      3
  ],
  "team": 0,
  "number": "12",
  "number_conf": 99.99838156485313,
  "appeared": 1,
  "disappeared": 897,
  "distance": 101.965,
  "frames_ball": 221,
  "%_ball": 0.24665178571428573
}...
```

# Usage
- Download ball and rim detection model file (https://drive.google.com/drive/folders/1cM7n1zpJ-KCk66g6NFRSK2X3PCII8hXK?usp=sharing)
- Modify main.py with your own basketball video path.
- Run main.py
- Input all necessary keypoints (court and scoreboard).
- Wait for completion...

# Explained
## Court tracking
This is being done quite simply. After you input all court keypoints in the first frame (like corners and line intersections), tool will track these keypoints (and all others that were not seen in the first frame) all throughout the video in few simple steps:
- SIFT algorithm places hundreds of points on current and previous frame in unique places (except on player bboxes and scoreboard bbox);
- FLANN algorithm finds exact pairs of points between these frames;
- Using the pairs, a homography matrix is calculated;
- Using the matrix court keypoints from previous frame are "moved" to current frame.

Court is being tracked, because:
1. Player tracking does not implement player classification from other people (like coaches, referees, fans etc.);
2. I wanted to know exact player location on the court.

## Player tracking
Player bboxes and segmentation polygons are found using the YOLOv8x model. Simply put tracking is being done by comparing player bbox IoU (intersection over union) in previous and current frame. Among other tests to improve tracking quality, tool includes team check and jersey number reading. Team check is being done by collecting all first frame player segmentation polygon histograms and clustering them in 2 classes using k-means. Jersey number reading is being done by utilizing EasyOCR.

## Ball and hoop tracking
For this a custom YOLOv8 model was trained, but the results where not good - hoop is being detected almost perfectly, but the ball is being detected very rarely, with some false positive detections on players heads and hands. This part definetly could be improved, as the initial plan was to track shots.

# Results
Could not find any annotated videos on which I could test the tool, but I did measure the time it takes to handle passed videos, as well as counted how many of players correctly detected. This was being done on an Apple MacBook Air M1 laptop and 6 unique 30 second basketball videos where used.
- One frame handling took about 1.5 seconds.
- Correct team assigned to player: 71%
- Correct number assigned to player: 57%
- Correct team and number: 44%

Certainly there is room for improvement...
