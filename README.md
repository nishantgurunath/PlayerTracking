# PlayerTracking
Use the below link to dowload the dataset and put the data in dataset/NCAA/ folder (create new folder)
http://www.cs.toronto.edu/~namdar/sports_course/ncaa/data.7z

Soccer Player detection reference
https://medium.com/@kananvyas/player-and-football-detection-using-opencv-python-in-fifa-match-6fd2e4e373f0

# Using detection module
For using detection, write this line at the start of your code
```bash
from detection.aniruddh.detector import detect
```
The input for this function is a BGR image and it outouts a N x 4 array of detection boxes
See example in main.py


