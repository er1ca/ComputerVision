import numpy as np
import cv2
import glob

a = 9
b = 6

#cap = cv2.VideoCapture('chessboard.avi')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
cv_gray_img = []

R = np.zeros((3,3),np.float32)
R[2,2] = 1.0

T = np.zeros((3,3),np.float32)
T[0] = (1,0,0)
T[1] = (0,1,0)
T[2] = (0,0,1)

# object points
objp = np.zeros((b*a,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FPS,60)
fps = cap.get(cv2.CAP_PROP_FPS)

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    # draw ground floor
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,255,255),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# calibration camera parameter
for img in glob.glob("calibration/*.png"):
    n = cv2.imread(img)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    #cv_img.append(n)
    cv_gray_img.append(n)

    ret, corners = cv2.findChessboardCorners(n,(a,b),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(a,b),(-1,-1),criteria)    
        objpoints.append(objp)
        imgpoints.append(corners2)

_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print(mtx, dist)


while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #find the chess board coarners
    ret, corners = cv2.findChessboardCorners(frame,(a,b),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(a,b),(-1,-1),criteria)  	

        ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)
       	
        #imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        rodRotMat = cv2.Rodrigues(rvecs)
        R[:3,:3] = rodRotMat[0]

        Rt = np.concatenate((R,tvecs),axis=1)        
        k=np.concatenate((axis,np.ones((8,1),np.float32)),axis=1)

        X = np.dot(Rt,k.T)
        projectedPoints = np.dot(mtx,X)
        projectedPoints = projectedPoints.T[:,:2]/np.repeat(projectedPoints.T[:,2],2).reshape(-1,2)
        
        
        frame = draw(frame, corners2, projectedPoints)
        #Draw and display the corners
        img = cv2.drawChessboardCorners(frame,(a,b), corners2,ret)
    
    cv2.imshow('img',frame)
    cv2.waitKey(500)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        print('quited')
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()