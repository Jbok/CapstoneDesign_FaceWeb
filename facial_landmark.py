# USAGE
# python3 facial_landmark.py --shape-predictor shape_predictor_68_face_landmarks.dat --image twotwo.jpg 

# Using Dlib & iBUG300-W dataset

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
import base64
from enum import IntEnum
from datetime import datetime

class Ptr(IntEnum):
   X = 0
   Y = 1

def cal_degrees(p1, p2,shape):
   # Calculate the degrees of two points
   width = abs(shape[p1][0]-shape[p2][0])
   height = abs(shape[p1][1]-shape[p2][1])

   if height == 0:
      return 90
	
   if width == 0:
      return 0
      	
   return math.degrees(math.atan(float(width)/float(height)))

def lips_average(shape):
   if abs(cal_degrees(27,54,shape)-cal_degrees(27,48,shape)) >= 10 or abs((cal_degrees(27,52,shape)-cal_degrees(27,50,shape))) >= 10:
      return -1
   else:
      return ((abs(cal_degrees(27,54,shape)-cal_degrees(27,48,shape)))+abs((cal_degrees(27,52,shape)-cal_degrees(27,50,shape))))/2

def nose_average(shape):
   if abs(cal_degrees(27,29,shape)) >= 10:
      return -1
   else:   
      return (abs(cal_degrees(27,29,shape)))

def facialline_average(shape):
   sum = 0
   for temp1 in range(4,8,1):
    if abs(cal_degrees(27,temp1,shape)-cal_degrees(27,16-temp1,shape)) >= 10:
      return -1
    else:
      sum += abs(cal_degrees(27,temp1,shape)-cal_degrees(27,16-temp1,shape))
   return (sum/4)

def draw_lines(image, shape):
   for temp in range(48,55,2):
      cv2.line(image, (shape[temp][0],shape[temp][1]), (shape[27][0],shape[27][1]), (255, 0, 0), 1)
      cv2.line(image, (shape[27][0],shape[27][1]), (shape[29][0],shape[29][1]), (255, 0, 0), 1)
      cv2.line(image, (shape[27][0],shape[27][1]), (shape[27][0],shape[30][1]), (255, 0, 0), 1)     
   for temp in range(1,8,1):
      cv2.line(image, (shape[temp][0],shape[temp][1]), (shape[27][0],shape[27][1]), (255, 0, 0), 1)
      cv2.line(image, (shape[16-temp][0],shape[16-temp][1]), (shape[27][0],shape[27][1]), (255, 0, 0), 1)


def cal_asymmetry(trained_data, image_path):
   # initialize dlib's face detector (HOG-based) and then create
   # the facial landmark predictor
   detector = dlib.get_frontal_face_detector()
   predictor = dlib.shape_predictor(trained_data)

   # load the input image, resize it, and convert it to grayscale
   image = cv2.imread(image_path)
   image = imutils.resize(image, width=300)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
   # detect faces in the grayscale image
   rects = detector(gray, 1)

   # Make 2D array
   points = [[0]*2 for i in range(200)]

   flag = 0

   # loop over the face detections
   if len(rects) >= 1:
      # loop over the face detections
      for (i, rect) in   enumerate(rects):
         shape = predictor(gray, rect)
         shape = face_utils.shape_to_np(shape)

      # convert dlib's rectangle to a OpenCV-style bounding box
      # [i.e., (x, y, w, h)], then draw the face bounding box
         (x, y, w, h) = face_utils.rect_to_bb(rect)
      
      # Print Rectangle 
         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # show the face number
         cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      # Print Rect Num
         #  print("Face #{}".format(i + 1))   

      # Point Number
         point = 0

         for (x, y) in shape:
            cv2.putText(image, "{}".format(point), (x+2,y+2),
               cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
            point = point + 1


         draw_lines(image, shape)
         jaw_degrees = facialline_average(shape)
         nose_degrees = nose_average(shape)
         lips_degrees = lips_average(shape)
         if jaw_degrees < 0:
            flag = -1

         if nose_degrees < 0:
            flag = -1

         if lips_degrees < 0:
            flag = -1


      cv2.imwrite('/tmp/tempLandmark.jpg', image)
      base64Data = ""
      with open('/tmp/tempLandmark.jpg', 'rb') as imageFile:
         base64Data = base64.b64encode(imageFile.read())
      base64Str = base64Data.decode('utf-8')

      if len(rects) > 1:
         return [-2, {'date': "-", 'jaw': "-", 'nose': "-", "lips" : "-"}, {"imageBase64": base64Str}]
      else:
         if flag == -1:
            return [-3, {'date': "-", 'jaw': "-", 'nose': "-", "lips" : "-"}, {"imageBase64": base64Str}]
         else:      
            return [0, {'date':datetime.today().strftime("%Y/%m/%d"), 'jaw':format(jaw_degrees, ".2f"), 'nose':format(nose_degrees, ".2f"), 'lips':format(lips_degrees, ".2f")}, {'imageBase64': base64Str}]

   elif len(rects) < 1:
      cv2.imwrite('/tmp/tempLandmark.jpg', image)
      base64Data = ""
      with open('/tmp/tempLandmark.jpg', 'rb') as imageFile:
         base64Data = base64.b64encode(imageFile.read())
      base64Str = base64Data.decode('utf-8')
      return [-1, {'date': "-", 'jaw': "-", 'nose': "-", "lips" : "-"}, {"imageBase64": base64Str}]

