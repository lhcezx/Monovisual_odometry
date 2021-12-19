import numpy as np 
import cv2
import pdb
import time
import numpy as np
from utils import *

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref, rgb = None):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1], currFeatures, status, track_error
	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]      # get the useful keypoint in last frame
	kp2 = kp2[st == 1]		   # get the useful keypoint in new frame
	# visualize_flux(rgb, kp1, kp2)

	return kp1, kp2


class PinholeCamera:
	def __init__(self, fx, fy, cx, cy):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy


class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		# self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.detector = cv2.SIFT_create()
		# self.detector = cv2.ORB_create(500, nlevels=15)

	def get_rgb_img(self, img):
		self.img_rgb = img


	# Calculate the key points of the image
	def processFirstFrame(self):
		start = time.perf_counter()
		self.px_ref = self.detector.detect(self.new_frame)

		# Shi-thomasi
		# self.px_ref = cv2.goodFeaturesToTrack(self.new_frame,5000,0.01,10)
		# self.px_ref = np.int0(self.px_ref)
		# for i in self.px_ref:
		# 	x,y = i.ravel()
		# 	cv2.circle(self.img_rgb,(x,y),3,255,-1)
		# img = cv2.drawKeypoints(self.img_rgb, self.px_ref, self.img_rgb)
		# cv2.imshow("img",self.img_rgb)
		# cv2.waitKey(0)
		# end = time.perf_counter()
		# print(str(end-start))
		self.px_ref = cv2.KeyPoint_convert(self.px_ref)   				# Conver to array
		self.frame_stage = STAGE_SECOND_FRAME


	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=0.001)    # return essential Matrix
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp) 							# Recovers the relative camera rotation and the translation from an estimated essential matrix and the corresponding points in two images
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur


	def processFrame(self, frame_id):
		scale = 10.0
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref, self.img_rgb)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
		self.cur_R = R.dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur


	def update(self, img, frame_id):
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
