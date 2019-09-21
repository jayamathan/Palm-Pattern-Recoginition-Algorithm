import Tkinter as Tk
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

img1 = cv2.imread('dataset2\chiran1.jpg',cv2.IMREAD_COLOR)
img2 = cv2.imread('dataset1\chiran1.jpg',cv2.IMREAD_COLOR)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#SIFT
detector = cv2.SIFT()

keypoints1 = detector.detect(gray1, None)
keypoints2 = detector.detect(gray2, None)

outimg1 = cv2.drawKeypoints(gray1, keypoints1)
outimg2 = cv2.drawKeypoints(gray2, keypoints2)

cv2.imshow('img1', outimg1)
cv2.imshow('img2', outimg2)

#kp,des = sift.compute(gray,kp)
kp1, des1 = detector.compute(gray1, keypoints1)
kp2, des2 = detector.compute(gray2, keypoints2)

print kp1
print kp2
print des1
print des2

matcher = cv2.BFMatcher()
matches = matcher.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

print matches
print "Length of matches :",len(matches)

def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        print "length of x1",x1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    #Show the image
    #cv2.imshow('Matched Features',out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')
    
    # Also return the image if you'd like a copy

    return out

end_img = drawMatches(img1, kp1, img2, kp2, matches[:75])
cv2.imshow('end_img', end_img)
cv2.waitKey(100)

#cv2.destroyAllWindows()



img1 = cv2.imread("dataset2\chiran1.jpg")
img2 = cv2.imread("dataset1\chiran1.png")
        
img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img12 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

imageA = cv2.resize(img11, (100, 100))
imageB = cv2.resize(img12, (100, 100))

s = ssim(imageA, imageB)

fig = plt.figure("_Result_")
if s<0:
    s=0
plt.suptitle("Percentage : %.2f " % (s*100))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(imageA, cmap = plt.cm.gray)
#plt.axis("off")

ax = fig.add_subplot(1, 2, 2)
plt.imshow(imageB, cmap = plt.cm.gray)
#plt.axis("off")

plt.show()
