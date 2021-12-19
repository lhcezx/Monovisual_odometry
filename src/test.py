import numpy as np 
import cv2
from calibration import *
from utils import matrix2val
from config import parse_configs
from visual_odometry import PinholeCamera, VisualOdometry
import datetime

configs = parse_configs()

_, mtx, dist, _, _ = calib(configs.checkerboard, configs.calib_dir)
fx, fy, cx, cy = matrix2val(mtx)
fx*=configs.scale
cx*=configs.scale
fy*=configs.scale
cy*=configs.scale


cam = PinholeCamera(fx, fy, cx, cy)
vo = VisualOdometry(cam)

# traj = np.full((600,600,3), 200, dtype=np.uint8)
traj = np.zeros((600,600,3), dtype=np.uint8)

# 5817, 5918
# 6451, 6544
for img_id in range(5817, 5918, 5):
	img = cv2.imread(configs.image_dir + str(img_id)+'.jpg')
	img = cv2.resize(img, [int(configs.orig_size[0]*configs.scale), int(configs.orig_size[1]*configs.scale)], interpolation= cv.INTER_AREA)
	vo.get_rgb_img(img)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	vo.update(img_gray, img_id)

	cur_t = vo.cur_t
	if(img_id > 5817):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+290, int(z)+90

	cv2.circle(traj, (draw_x,draw_y), 2, (img_id*255/4540,255-img_id*255/4540,0), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
	text = "Coordinates: x=%2f y=%2f z=%2f"%(x,y,z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	cv2.namedWindow('Road facing camera', cv2.WINDOW_NORMAL)
	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)


cv2.imwrite('results/map.png', traj)
