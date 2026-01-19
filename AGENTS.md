This folder contains python scripts to help vision AI dataset preparation. Typically there are images in the default folder "images" and also can be corresponding annotation labels in YOLO detection or YOLO pose format.



When creating or updating a python script, also maintain its .md description file. Include new scripts into README.md file.



Python scripts, by default, at the start printing what they are going to do, listing all parameters; show progress bar with estimated completion time like "ETA 14:47"; print stats at the end.



Python script distinguishes between Current Working Directory (typically contains folders and files to process) and Script Directory (typically contains script, description, configuration file and pytorch model files .pt).



By default, YOLO pose format contains 17 keypoints; see file dataset.yaml for the shape, names, flip ids.


mpii-human-pose keypoint names and ids: 0 - right ankle, 1 - right knee, 2 - right hip, 3 - left hip, 4 - left knee, 5 - left ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - right wrist, 11 - right elbow, 12 - right shoulder, 13 - left shoulder, 14 - left elbow, 15 - left wrist.

