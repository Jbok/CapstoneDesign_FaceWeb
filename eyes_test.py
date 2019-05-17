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
         if point == 27:
            print("nose-front of eyes \n")
            cv2.line(image, (x, y), (shape[39][0], shape[39][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.39 :",cal_distances(x,y,shape[39][0],shape[39][1]))
            p27_39_dist = cal_distances(x,y,shape[39][0],shape[39][1])
            print("Degree between p.{}".format(point), "and p.39 :", cal_degrees(x,y,shape[39][0],shape[39][1]), "\n")
            p27_39_deg = cal_degrees(x,y,shape[39][0],shape[39][1])

            cv2.line(image, (x, y), (shape[42][0], shape[42][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.42 :",cal_distances(x,y,shape[42][0],shape[42][1]))
            p27_42_dist = cal_distances(x,y,shape[42][0],shape[42][1])
            print("Degree between p.{}".format(point), "and p.42 :", cal_degrees(x,y,shape[42][0],shape[42][1]), "\n")
            p27_42_deg = cal_degrees(x,y,shape[42][0],shape[42][1])

            cv2.line(image, (x, y), (shape[36][0], shape[36][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.36 :",cal_distances(x,y,shape[36][0],shape[36][1]))
            p27_36_dist = cal_distances(x,y,shape[36][0],shape[36][1])
            print("Degree between p.{}".format(point), "and p.36 :", cal_degrees(x,y,shape[36][0],shape[36][1]), "\n")
            p27_36_deg = cal_degrees(x,y,shape[36][0],shape[36][1])
            
            cv2.line(image, (x, y), (shape[45][0], shape[45][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.45 :",cal_distances(x,y,shape[45][0],shape[45][1]))
            p27_45_dist = cal_distances(x,y,shape[45][0],shape[45][1])
            print("Degree between p.{}".format(point), "and p.45 :", cal_degrees(x,y,shape[45][0],shape[45][1]), "\n")
            p27_45_deg = cal_degrees(x,y,shape[45][0],shape[45][1])

         if point == 28:
            print("nose-front of eyes \n")
            cv2.line(image, (x, y), (shape[39][0], shape[39][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.39 :",cal_distances(x,y,shape[39][0],shape[39][1]))
            p28_39_dist = cal_distances(x,y,shape[39][0],shape[39][1])
            print("Degree between p.{}".format(point), "and p.39 :", cal_degrees(x,y,shape[39][0],shape[39][1]), "\n")
            p28_39_deg = cal_degrees(x,y,shape[39][0],shape[39][1])

            cv2.line(image, (x, y), (shape[42][0], shape[42][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.42 :",cal_distances(x,y,shape[42][0],shape[42][1]))
            p28_42_dist = cal_distances(x,y,shape[42][0],shape[42][1])
            print("Degree between p.{}".format(point), "and p.42 :", cal_degrees(x,y,shape[42][0],shape[42][1]), "\n")
            p28_42_deg = cal_degrees(x,y,shape[42][0],shape[42][1])

            cv2.line(image, (x, y), (shape[36][0], shape[36][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.36 :", cal_distances(x,y,shape[36][0],shape[36][1]))
            p28_36_dist = cal_distances(x,y,shape[36][0],shape[36][1])
            print("Degree between p.{}".format(point), "and p.36 :", cal_degrees(x,y,shape[36][0],shape[36][1]), "\n")
            p28_36_deg = cal_degrees(x,y,shape[36][0],shape[36][1])

            cv2.line(image, (x, y), (shape[45][0], shape[45][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.45 :",cal_distances(x,y,shape[45][0],shape[45][1]))
            p28_45_dist = cal_distances(x,y,shape[45][0],shape[45][1])
            print("Degree between p.{}".format(point), "and p.45 :", cal_degrees(x,y,shape[45][0],shape[45][1]), "\n")
            p28_45_deg = cal_degrees(x,y,shape[45][0],shape[45][1])      
         
         if point == 29:
            print("nose-front of eyes \n")
            cv2.line(image, (x, y), (shape[39][0], shape[39][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.39 :",cal_distances(x,y,shape[39][0],shape[39][1]))
            p29_39_dist = cal_distances(x,y,shape[39][0],shape[39][1])
            print("Degree between p.{}".format(point), "and p.39 :", cal_degrees(x,y,shape[39][0],shape[39][1]), "\n")
            p29_39_deg = cal_degrees(x,y,shape[39][0],shape[39][1])
            
            cv2.line(image, (x, y), (shape[42][0], shape[42][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.42 :",cal_distances(x,y,shape[42][0],shape[42][1]))
            p29_42_dist = cal_distances(x,y,shape[42][0],shape[42][1])
            print("Degree between p.{}".format(point), "and p.42 :", cal_degrees(x,y,shape[42][0],shape[42][1]), "\n")
            p29_42_deg = cal_degrees(x,y,shape[42][0],shape[42][1])
            
            cv2.line(image, (x, y), (shape[36][0], shape[36][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.36 :",cal_distances(x,y,shape[36][0],shape[36][1]))
            p29_36_dist = cal_distances(x,y,shape[36][0],shape[36][1])
            print("Degree between p.{}".format(point), "and p.36 :", cal_degrees(x,y,shape[36][0],shape[36][1]), "\n")
            p29_36_deg = cal_degrees(x,y,shape[36][0],shape[36][1])

            cv2.line(image, (x, y), (shape[45][0], shape[45][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.45 :",cal_distances(x,y,shape[45][0],shape[45][1]))
            p29_45_dist = cal_distances(x,y,shape[45][0],shape[45][1])
            print("Degree between p.{}".format(point), "and p.45 :", cal_degrees(x,y,shape[45][0],shape[45][1]), "\n")
            p29_45_deg = cal_degrees(x,y,shape[45][0],shape[45][1])
      
         if point == 30:
            print("nose-front of eyes \n")
            cv2.line(image, (x, y), (shape[39][0], shape[39][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.39 :",cal_distances(x,y,shape[39][0],shape[39][1]))
            p30_39_dist = cal_distances(x,y,shape[39][0],shape[39][1])
            print("Degree between p.{}".format(point), "and p.39 :", cal_degrees(x,y,shape[39][0],shape[39][1]), "\n")
            p30_39_deg = cal_degrees(x,y,shape[39][0],shape[39][1])

            cv2.line(image, (x, y), (shape[42][0], shape[42][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.42 :",cal_distances(x,y,shape[42][0],shape[42][1]))
            p30_42_dist = cal_distances(x,y,shape[42][0],shape[42][1])
            print("Degree between p.{}".format(point), "and p.42 :", cal_degrees(x,y,shape[42][0],shape[42][1]), "\n")
            p30_42_deg = cal_degrees(x,y,shape[42][0],shape[42][1])

            cv2.line(image, (x, y), (shape[36][0], shape[36][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.36 :",cal_distances(x,y,shape[36][0],shape[36][1]))
            p30_36_dist = cal_distances(x,y,shape[36][0],shape[36][1])
            print("Degree between p.{}".format(point), "and p.36 :", cal_degrees(x,y,shape[36][0],shape[36][1]), "\n")
            p30_36_deg = cal_degrees(x,y,shape[36][0],shape[36][1])

            cv2.line(image, (x, y), (shape[45][0], shape[45][1]), (255, 0, 0), 1)
            print("Distance between p.{}".format(point), "and p.45 :",cal_distances(x,y,shape[45][0],shape[45][1]))
            p30_45_dist = cal_distances(x,y,shape[45][0],shape[45][1])
            print("Degree between p.{}".format(point), "and p.45 :", cal_degrees(x,y,shape[45][0],shape[45][1]), "\n")      
            p30_45_deg = cal_degrees(x,y,shape[45][0],shape[45][1])




         # print Point
         # print ("p.{}".format(point),(x,y))
         point = point + 1


      return {'Distance between p.27 and p.39':p27_39_dist, 'Degree between p.27 and p.39':p27_39_deg, 
      'Distance between p.27 and p.42':p27_42_dist, 'Degree between p.27 and p.42':p27_42_deg,
      'Distance between p.27 and p.36':p27_36_dist, 'Degree between p.27 and p.36':p27_36_deg,
      'Distance between p.27 and p.45':p27_45_dist, 'Degree between p.27 and p.45':p27_45_deg,
      'Distance between p.28 and p.39':p28_39_dist, 'Degree between p.28 and p.39':p28_39_deg, 
      'Distance between p.28 and p.42':p28_42_dist, 'Degree between p.28 and p.42':p28_42_deg,
      'Distance between p.28 and p.36':p28_36_dist, 'Degree between p.28 and p.36':p28_36_deg,
      'Distance between p.28 and p.45':p28_45_dist, 'Degree between p.28 and p.45':p28_45_deg,
      'Distance between p.29 and p.39':p29_39_dist, 'Degree between p.29 and p.39':p29_39_deg, 
      'Distance between p.29 and p.42':p29_42_dist, 'Degree between p.29 and p.42':p29_42_deg,
      'Distance between p.29 and p.36':p29_36_dist, 'Degree between p.29 and p.36':p29_36_deg,
      'Distance between p.29 and p.45':p29_45_dist, 'Degree between p.29 and p.45':p29_45_deg,
      'Distance between p.30 and p.39':p30_39_dist, 'Degree between p.30 and p.39':p30_39_deg, 
      'Distance between p.30 and p.42':p30_42_dist, 'Degree between p.30 and p.42':p30_42_deg,
      'Distance between p.30 and p.36':p30_36_dist, 'Degree between p.30 and p.36':p30_36_deg,
      'Distance between p.30 and p.45':p30_45_dist, 'Degree between p.30 and p.45':p30_45_deg,}     