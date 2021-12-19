# Monovisual_odometry
The aim is to compare and briefly analyse the difference in performance of the visual odometer in day and night scenes. Place day and night sequences of images in the day and night folders respectively, then place the checkerboard grid calibration images in the Calibration folder to get camera intrinsics parameters. The method is the Lk optical flow method and the extrinsic parameter is obtained by 2D-2D epipolar geometry. 


## Folder structure

```
${ROOT}
└── results/
└── dataset/    
│   └── calibration/
│   │     
│   │    
│   │
│   ├── image/
|        ├── day/
|        ├── night/
|
├── src     
├── README.md 
```


### References
1. [monoVO-python](https://github.com/uoip/monoVO-python)<br>
2. [视觉SLAM 14讲](https://www.bilibili.com/video/BV16t411g7FR/)
