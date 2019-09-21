import numpy as np
import cv2
img=cv2.imread('m.png')

fgbg=cv2.BackgroundSubtractorMOG2()
fgbg=fgbg.apply(img)
cv2.imshow('fgmask',img)


cv2.waitkey(0)

cv2.destroyAllWindows()
 
