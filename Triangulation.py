import numpy as np 
import cv2 
from matplotlib import pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D

#웹캠
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

########### Initialization ############

a = 9
b = 6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
cv_gray_img = []

R1 = np.zeros((3,3),np.float32)
R1[2,2] = 1.0

R2 = np.zeros((3,3),np.float32)
R2[2,2] = 1.0

# 3d object points
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)


fc = 0
f_pts = []

folder = 'test1'

####################################

#카메라 캘리브레이션 
for img in glob.glob("calibration/*.png"):
    n = cv2.imread(img)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    cv_gray_img.append(n)

    ret, corners = cv2.findChessboardCorners(n,(a,b),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(a,b),(-1,-1),criteria)    
        objpoints.append(objp)
        imgpoints.append(corners2)

_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


##########        START          ##########

#  [1] 레퍼런스 이미지 저장
while (True):
	ret, frame = cap.read()		
		
	#현재 웹캠 이미지 출력
	cv2.imshow('img',frame)

	#'space bar' 입력 ==> 현재 이미지 프레임을 레퍼런스 이미지로 저장 
	if cv2.waitKey(1) & 0xFF == 32:
		img_counter = len(glob.glob(str(folder)+"/*.png"))
		img_counter += 1
		img_name = "{}.png".format(img_counter)
		cv2.imwrite(str(folder)+"/"+img_name, frame)
		print("{} written!".format(img_name))
		break

cv2.destroyAllWindows()


#  [2] 레퍼런스 이미지와 웹캠 이미지 프레임간에 특징점 매칭 

#레퍼런스 이미지 로드 
img1 = cv2.imread(str(folder)+'/'+str(img_name),0) 
print("** reference image loaded... "+str(img_name))

ret, corners = cv2.findChessboardCorners(img1,(a,b),None)
if ret:
	corners_r = cv2.cornerSubPix(img1,corners,(a,b),(-1,-1),criteria)

	ret, rvecs_r, tvecs_r, _ = cv2.solvePnPRansac(objp, corners_r, mtx, dist)

	rodRotMat_r = cv2.Rodrigues(rvecs_r)
	R1[:3,:3] = rodRotMat_r[0]
	refRT = np.concatenate((R1,tvecs_r),axis=1)
	P0 = np.dot(mtx, refRT)
		

#동영상 저장을 위한 초기화   ##############################
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width,frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoFile = str(folder)+'/'+str(img_counter)+'.avi'
out = cv2.VideoWriter(videoFile, fourcc, 20 ,size)


while (True):
	fc += 1
	print('frame #',fc)
		
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	

	#특징 추출 (ORB 사용)
	orb = cv2.ORB_create(nfeatures=1000)
	
	kp1, des1 = orb.detectAndCompute(img1,None) #keypoint, descriptor
	kp2, des2 = orb.detectAndCompute(frame,None)

	#특징점 매칭
	bf = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

	good = []
	for i,m in enumerate(matches):
		if i < len(matches)- 1 and m.distance < 0.8 * matches[i+1].distance:
			good.append(m)

	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
	
	#특징점 매칭 결과 출력
	img3 = cv2.drawMatches(img1,kp1,frame,kp2,matches[:500],None)
	

# [3] Triangulation으로 3D Position을 reconstruction

	#현재 프레임에서 체스보드 찾기 
	ret, found_f = cv2.findChessboardCorners(frame,(a,b),None)

	if ret == True:
		corners_f = cv2.cornerSubPix(gray,corners,(a,b),(-1,-1),criteria)

		#Rotation/Translation matrix
		ret, rvecs_f, tvecs_f, _ = cv2.solvePnPRansac(objp, corners_f, mtx, dist)
		rodRotMat_f = cv2.Rodrigues(rvecs_f)
		R2[:3,:3] = rodRotMat_f[0]
		frameRT = np.concatenate((R2,tvecs_f),axis=1)

		#Projection Matrix 계산
		P1 = np.dot(mtx,frameRT)
		
		# Triangulation으로 3D points reconsturciton
		pts4D = cv2.triangulatePoints(P0,P1,src_pts,dst_pts)

		#3D Points로 Reshape
		pts3D = pts4D.T[:,:3]/np.repeat(pts4D.T[:,3],3).reshape(-1,3)

		#Projection to image plane
		imgpts, jac = cv2.projectPoints(pts3D,rvecs_f,tvecs_f,mtx,dist)

		#2D Points Reshape
		imgpts = np.int32(imgpts).reshape(-1,2)
		
		for ip in imgpts:
			img4 = cv2.circle(frame,tuple(ip),1,(0,0,255),2)
		
	#현재 프레임 동영상 저장
		out.write(img4)

	#결과 출력 
	cv2.imshow('img',img3)
	
	k = cv2.waitKey(1)
	if k & 0xFF == ord('q'):
		print('quited')
		break
	elif k & 0xFF == 32:		
		break

cap.release()
out.release()
cv2.destroyAllWindows()


#저장한 동영상 로드 
print('Read the saved video file... ',videoFile)
cap2 = cv2.VideoCapture(videoFile)
while(cap2.isOpened()):
	ret, frame2 = cap2.read()


	if ret:
		cv2.imshow('video',frame2)

	#프레임이 끝나면 braek
	elif not ret:
		break

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

cap2.release()
cv2.destroyAllWindows()

