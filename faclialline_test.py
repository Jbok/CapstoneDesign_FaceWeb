# USAGE
# python fltest.py --shape-predictor shape_predictor_68_face_landmarks.dat --image example_01.jpg 

# Using Dlib & iBUG300-W dataset

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math


def cal_degrees(p1, p2, p3, p4):
    # Calculate the degrees of two points
    width = abs(p1-p3)
    height = abs(p2-p4)

    if height == 0:
        return 0
	
    if width == 0:
        return 90
	
    return math.degrees(math.atan(float(width)/float(height)))

def cal_distances(p1, p2, p3, p4):
   # Calculate the distances of two points
   a = p3-p1
   b = p4-p2 
   result = math.sqrt((a*a)+(b*b))
   return result

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
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # show the face number
      cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      # Print Rect Num
      print("Face #{}".format(i + 1))   

      # Point Number
      point = 0

      # loop over the (x, y)-coordinates for the facial landmarks
      # and draw them on the image
      # loop over the (x, y)-coordinates for the facial landmarks
      # and draw them on the image
        for (x, y) in shape:
         cv2.putText(image, "{}".format(point), (x+2,y+2),
         cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
         cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

  

         # Print nose-lip distance calculation     
         if point == 30:
            cv2.line(image, (x, y), (shape[0][0], shape[0][1]), (255, 0, 0), 1)
            p30_0_dist = cal_distances(x,y,shape[0][0],shape[0][1])
            p30_0_deg = cal_degrees(x,y,shape[0][0],shape[0][1])

            cv2.line(image, (x, y), (shape[16][0], shape[16][1]), (255, 0, 0), 1)
            p30_16_dist = cal_distances(x,y,shape[16][0],shape[16][1])
            p30_16_deg = cal_degrees(x,y,shape[16][0],shape[16][1])

            cv2.line(image, (x, y), (shape[1][0], shape[1][1]), (255, 0, 0), 1)
            p30_1_dist = cal_distances(x,y,shape[1][0],shape[1][1])
            p30_1_deg = cal_degrees(x,y,shape[1][0],shape[1][1])

            cv2.line(image, (x, y), (shape[15][0], shape[15][1]), (255, 0, 0), 1)
            p30_15_dist = cal_distances(x,y,shape[15][0],shape[15][1])
            p30_15_deg = cal_degrees(x,y,shape[15][0],shape[15][1])

            cv2.line(image, (x, y), (shape[2][0], shape[2][1]), (255, 0, 0), 1)
            p30_2_dist = cal_distances(x,y,shape[2][0],shape[2][1])
            p30_2_deg = cal_degrees(x,y,shape[2][0],shape[2][1])

            cv2.line(image, (x, y), (shape[14][0], shape[14][1]), (255, 0, 0), 1)
            p30_14_dist = cal_distances(x,y,shape[14][0],shape[14][1])
            p30_14_deg = cal_degrees(x,y,shape[14][0],shape[14][1])

            cv2.line(image, (x, y), (shape[3][0], shape[3][1]), (255, 0, 0), 1)
            p30_3_dist = cal_distances(x,y,shape[3][0],shape[3][1])
            p30_3_deg = cal_degrees(x,y,shape[3][0],shape[3][1])

            cv2.line(image, (x, y), (shape[13][0], shape[13][1]), (255, 0, 0), 1)
            p30_13_dist = cal_distances(x,y,shape[13][0],shape[13][1])
            p30_13_deg = cal_degrees(x,y,shape[13][0],shape[13][1])

            cv2.line(image, (x, y), (shape[4][0], shape[4][1]), (255, 0, 0), 1)
            p30_4_dist = cal_distances(x,y,shape[4][0],shape[4][1])
            p30_4_deg = cal_degrees(x,y,shape[4][0],shape[4][1])

            cv2.line(image, (x, y), (shape[12][0], shape[12][1]), (255, 0, 0), 1)
            p30_12_dist = cal_distances(x,y,shape[12][0],shape[12][1])
            p30_12_deg = cal_degrees(x,y,shape[12][0],shape[12][1])

            cv2.line(image, (x, y), (shape[5][0], shape[5][1]), (255, 0, 0), 1)
            p30_5_dist = cal_distances(x,y,shape[5][0],shape[5][1])
            p30_5_deg = cal_degrees(x,y,shape[5][0],shape[5][1])

            cv2.line(image, (x, y), (shape[11][0], shape[11][1]), (255, 0, 0), 1)
            p30_11_dist = cal_distances(x,y,shape[11][0],shape[11][1])
            p30_11_deg = cal_degrees(x,y,shape[11][0],shape[11][1])

            cv2.line(image, (x, y), (shape[6][0], shape[6][1]), (255, 0, 0), 1)
            p30_6_dist = cal_distances(x,y,shape[6][0],shape[6][1])
            p30_6_deg = cal_degrees(x,y,shape[6][0],shape[6][1])

            cv2.line(image, (x, y), (shape[10][0], shape[10][1]), (255, 0, 0), 1)
            p30_10_dist = cal_distances(x,y,shape[10][0],shape[10][1])
            p30_10_deg = cal_degrees(x,y,shape[10][0],shape[10][1])

            cv2.line(image, (x, y), (shape[7][0], shape[7][1]), (255, 0, 0), 1)
            p30_7_dist = cal_distances(x,y,shape[7][0],shape[7][1])
            p30_7_deg = cal_degrees(x,y,shape[7][0],shape[7][1])

            cv2.line(image, (x, y), (shape[9][0], shape[9][1]), (255, 0, 0), 1)
            p30_9_dist = cal_distances(x,y,shape[9][0],shape[9][1])
            p30_9_deg = cal_degrees(x,y,shape[9][0],shape[9][1])


            


         # print Point
         # print ("p.{}".format(point),(x,y))
         point = point + 1


      return {'Distance between p.30 and p.0':p30_0_dist, 'Degree between p.30 and p.0':p30_0_deg, 
      'Distance between p.30 and p.16':p30_16_dist, 'Degree between p.30 and p.16':p30_16_deg,
      'Distance between p.30 and p.1':p30_1_dist, 'Degree between p.30 and p.50':p30_1_deg,
      'Distance between p.30 and p.15':p30_15_dist, 'Degree between p.30 and p.52':p30_15_deg
      'Distance between p.30 and p.2':p30_2_dist, 'Degree between p.30 and p.16':p30_2_deg,
      'Distance between p.30 and p.14':p30_14_dist, 'Degree between p.30 and p.50':p30_14_deg,
      'Distance between p.30 and p.3':p30_3_dist, 'Degree between p.30 and p.52':p30_3_deg,
      'Distance between p.30 and p.13':p30_16_dist, 'Degree between p.30 and p.16':p30_13_deg,
      'Distance between p.30 and p.4':p30_4_dist, 'Degree between p.30 and p.50':p30_4_deg,
      'Distance between p.30 and p.12':p30_12_dist, 'Degree between p.30 and p.52':p30_12_deg,
      'Distance between p.30 and p.5':p30_5_dist, 'Degree between p.30 and p.16':p30_5_deg,
      'Distance between p.30 and p.11':p30_11_dist, 'Degree between p.30 and p.50':p30_11_deg,
      'Distance between p.30 and p.6':p30_6_dist, 'Degree between p.30 and p.52':p30_6_deg,
      'Distance between p.30 and p.10':p30_10_dist, 'Degree between p.30 and p.16':p30_10_deg,
      'Distance between p.30 and p.7':p30_7_dist, 'Degree between p.30 and p.50':p30_7_deg,
      'Distance between p.30 and p.9':p30_9_dist, 'Degree between p.30 and p.52':p30_9_deg}     
