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
<<<<<<< HEAD
=======
from enum import IntEnum
from datetime import datetime

class Ptr(IntEnum):
    X = 0
    Y = 1
>>>>>>> sls

def cal_degrees(p1, p2,shape):
    # Calculate the degrees of two points
    width = abs(shape[p1][0]-shape[p2][0])
    height = abs(shape[p1][1]-shape[p2][1])

    if height == 0:
        return 0
	
    if width == 0:
        return 90
	
<<<<<<< HEAD
    return math.degrees(math.atan(float(width)/float(height)))

def lips_average(shape):
  return ((abs(cal_degrees(27,54,shape)-cal_degrees(27,48,shape)))+abs((cal_degrees(27,52,shape)-cal_degrees(27,50,shape))))/2

def nose_average(shape):
   return (abs(cal_degrees(27,30,shape)))

def facialline_average(shape):
   sum = 0
   for temp1 in range(1,8,1):
    sum += abs(cal_degrees(27,temp1,shape)-cal_degrees(27,16-temp1,shape))
   return (sum/8)


def cal_asymmetry(rects):
   if len(rects) == 1:
         # loop over the face detections
      for (i, rect) in enumerate(rects):
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

         jaw_degrees = facialline_average(shape)
         nose_degrees = nose_average(shape)
         lips_degrees = lips_average(shape)
         print("jaw: ", jaw_degrees, "nose: ",nose_degrees, "lips: ",lips_degrees)
         return {'jaw':jaw_degrees, 'nose':nose_degrees, 'lips':lips_degrees}
   elif len(rects) < 1 :
      return -1   
   elif len(rects) > 1 :
      return -2   

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
   help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
   help="path to input image")
args = vars(ap.parse_args())

   # initialize dlib's face detector (HOG-based) and then create
   # the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

   # load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
   #image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # detect faces in the grayscale image
rects = detector(gray, 1) 
cal_asymmetry(rects)


cv2.imshow("Output", image)
cv2.imwrite('iu2.jpg', image)
cv2.waitKey(0)
=======
    return math.degrees(math.atan(float(height)/float(width)))

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


    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        # Print Rectangle 
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print Rect Num
        # print("Face #{}".format(i + 1))	

        # Point Number
        point_num = 0

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            points[point_num][Ptr.X] = x
            points[point_num][Ptr.Y] = y

        #     # print Point
        #     print ("p.{}".format(point),(x,y))
            point_num = point_num + 1

        #     cv2.putText(image, "{}".format(point), (x+2,y+2),
        #         cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
        #     cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        #     # Print Center Line
        #     if point == 28:
        #         cv2.line(image, (x, image.shape[0]), (x, 0), (255, 0, 0), 1)

        #     # Draw Line to Jaw
        #     if point <= 8:
        #         cv2.line(image, (x, y), (shape[16-point][0], shape[16-point][1]), (255, 0, 0), 1)


        #     # Draw Line to eyes
        #     if point == 36:
        #         cv2.line(image, (x, y), (shape[45][0], shape[45][1]), (255, 0, 0), 1)
        #     if point == 37:
        #         cv2.line(image, (x, y), (shape[44][0], shape[44][1]), (255, 0, 0), 1)
        #     if point == 38:
        #         cv2.line(image, (x, y), (shape[43][0], shape[43][1]), (255, 0, 0), 1)
        #     if point == 39:
        #         cv2.line(image, (x, y), (shape[42][0], shape[42][1]), (255, 0, 0), 1)
        #     if point == 40:
        #         cv2.line(image, (x, y), (shape[47][0], shape[47][1]), (255, 0, 0), 1)
        #     if point == 41:
        #         cv2.line(image, (x, y), (shape[46][0], shape[46][1]), (255, 0, 0), 1)



        jaw_degrees = 90 - cal_degrees(points[9], points[28])
				
        eyebrow_degrees = cal_degrees(points[20], points[25])
				
        lips_degrees = cal_degrees(points[49], points[55])

        return {'date':datetime.today().strftime("%Y/%m/%d"), 'jaw':format(jaw_degrees, ".2f"), 'eye':format(eyebrow_degrees, ".2f"), 'lips':format(lips_degrees, ".2f")}
>>>>>>> sls
