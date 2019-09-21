import Tkinter as Tk
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2




img1 = cv2.imread("h1.png")
img2 = cv2.imread("h2.png")
        
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
