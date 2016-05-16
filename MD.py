import cv2
import numpy as np
from preProcessing import PP

div = 400

def resize(img):
  RP = img.shape[0] / div if img.shape[0] <= img.shape[1] else img.shape[1] / div   # Setting resizing Parameter keeping aspect ratio constant
  imgResized = cv2.resize (img, (img.shape[1] / RP, img.shape[0] / RP))
  return imgResized

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  motion = cv2.bitwise_and(d1, d2)
  # print motion
  obj = PP(motion)
  motionInBlack = obj.preprocess()
  # motion = cv2.medianBlur(motion, 5)
  # motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
  # # motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  # # motion = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY_INV)[1]
  # thresh, motion = cv2.threshold(motion, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
  # print thresh
  motionCopy = motionInBlack.copy()
  cv2.imshow("moving Objects", motionInBlack)
  contours, hiearchy = cv2.findContours(motionInBlack, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(motionCopy, contours, -1, (0, 0, 0), 2)
  cv2.imshow("Moving object contours in Black", motionCopy)
  return cv2.bitwise_and(d1, d2)


def mD(cam):

  # cv2.namedWindow("Movement Indicator", cv2.CV_WINDOW_AUTOSIZE)

  # Read three images first:
  t_minus = resize(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY))
  t = resize(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY))
  t_plus = resize(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY))

  while cam.read()[0]:
    cv2.imshow( "Movement Indicator", diffImg(t_minus, t, t_plus) )

    # Read next image
    t_minus = t
    t = t_plus
    t_plus = resize(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY))

    if cv2.waitKey(1) & 0xFF == 27:
      cv2.destroyAllWindows
      break

cam = cv2.VideoCapture("sample2.mp4")
mD(cam)