import cv2
#import imutils
#import argparse
import os
import face_recognition
import datetime
import pickle
import numpy as np
# from mtcnn.mtcnn import MTCNN
#from keras import backend as K

def compareFaceWithDataset (listOfFaceEncodingsKnown, faceEncodingToCompare, similarityThreshold = 0.6):
	distanceResult = np.empty((0))
	if (len(listOfFaceEncodingsKnown) != 0):
		distanceResult = np.linalg.norm(listOfFaceEncodingsKnown - faceEncodingToCompare, axis = 1)
	print (distanceResult)
	return list((distanceResult) <= similarityThreshold)

def getNumberOfPersonInDataset (nameToCount, data):
     count = 0
     for name in data["names"]:
          if (name == nameToCount):
               count += 1
     return count

def createFolder (path):
     if os.path.exists(path) == False:
          os.mkdir(path)

def saveDataToPickleFile (knownNames,knownEncodings,pathToSave):
     print ("Saving Encodings to Pickle file ...!")
     data = {"names":knownNames, "encodings":knownEncodings}
     f = open(pathToSave,"wb")
     f.write(pickle.dumps(data))
     f.close()

def loadDataFromPickleFile (pathToSave):
	print ("Loading Encodings from Pickle file ...!") 
	f = open(pathToSave,"rb")
	data = pickle.loads(f.read())
	f.close()
	return data

def detect (image, method): #0: MTCNN, 1: haar-cascade, 2: hog
     boxOfFaces = []
     time = datetime.datetime.now()
     # if (method == "MTCNN"):
     #      print ("dung mtcnn")
     #      image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
     #      detector = MTCNN()
     #      faces = detector.detect_faces(image_rgb)
     #      for face in faces:
     #           print ("confidence:", face["confidence"])
     #           boxDetect = face["box"] #box = (x,y-left,top width-height)
     #           left, top, right, bottom = abs(boxDetect[0]), abs(boxDetect[1]), abs(boxDetect[0] + boxDetect[2]), abs(boxDetect[1] + boxDetect[3])
     #           boxDetect = (top,right,bottom,left)
     #           boxOfFaces.append(boxDetect)
     if (method == "Haar-Cascade"):
          print ("use haar-cascade")
          base = os.path.dirname(os.path.abspath(__file__))
          pathToCascadeXML = os.path.join(base,"static","haarcascade_frontalface_default.xml")
          face_cascade = cv2.CascadeClassifier(pathToCascadeXML)
          image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors=3, minSize=(50,50))
          for face in faces:
               left, top, width, height = face
               right = left + width
               bottom = top + height
               boxDetect = (top, right, bottom, left)
               boxOfFaces.append(boxDetect)
     else: #method = HOG
          print ("use hog")
          image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          boxes = face_recognition.face_locations(image_rgb, model="hog")
          for boxDetect in boxes:
               boxOfFaces.append(boxDetect)
     time_detection = (datetime.datetime.now() - time).total_seconds()
     return (time_detection, boxOfFaces)

def encode (image, boxOfFaces):
     image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
     time = datetime.datetime.now()
     encodings = face_recognition.face_encodings(image_rgb, boxOfFaces)
     time_embedding = (datetime.datetime.now() - time).total_seconds()
     return (time_embedding, encodings)

def match (image, encodings, data, thresholdConfidence = 30.0):
     names = []
     confidences = []
     time = datetime.datetime.now()
     for encoding in encodings:
          # compareResult = face_recognition.compare_faces(data["encodings"],encoding)
          compareResult = compareFaceWithDataset(data["encodings"],encoding)
          name = "Unknown"
          confidence = 100.0
          if True in compareResult:
               listPeople = {}
               for index,match in enumerate(compareResult):
                    if match:
                         n = data["names"][index]
                         listPeople[n] = listPeople.get(n,0) + 1
               for n in listPeople:
                    listPeople[n] = round(1.0 * listPeople[n] * 100 / getNumberOfPersonInDataset(n,data),2)
               if max(listPeople.values()) > thresholdConfidence:
                    name = max(listPeople, key=listPeople.get)
                    confidence = listPeople[name]
          names.append(name)
          confidences.append(confidence)
     time_recognition = (datetime.datetime.now() - time).total_seconds()
     return (time_recognition, names, confidences)

def faceRecognition (pathToImage, pathToSave, method_detect = 0):
     pathToSaveEncoding = "/home/vnpt/facialRecognition/data/dataset.pickle"
     data = loadDataFromPickleFile(pathToSaveEncoding)
     image = cv2.imread(pathToImage)
     timeDetection, boxes = detect(image,method_detect)
     timeEncoding, encodings = encode(image,boxes)
     timeRecognition, names, confidences = match(image,encodings,data)
     for index,name in enumerate(names):
          top,right,bottom,left = boxes[index]
          cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 4)
          text = name + ":{}%".format(confidences[index])
          cv2.putText(image,text,(left,bottom + 30),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4,cv2.LINE_AA)

     print ("Saving ",pathToSave)
     cv2.imwrite(pathToSave, image)
     #K.clear_session()
     print (timeDetection,timeEncoding,timeRecognition)
     return (round(timeDetection,4),round(timeEncoding,4),round(timeRecognition,4))

def initiate ():    #this fuction to make face dataset and encoding for recognition
     basePeople = "/home/vnpt/facialRecognition/data/people/"
     baseFace = "/home/vnpt/facialRecognition/data/faces/"
     pathToSaveEncoding = "/home/vnpt/facialRecognition/data/dataset.pickle"
     padding = 0
     knownNames = []
     knownEncodings = []
     for path, subdir, files in os.walk(basePeople):
          if files:
               count = 0
               namePerson = path.split(os.sep)[-1]
               pathToFaceFolder = os.path.join(baseFace,namePerson)
               createFolder(pathToFaceFolder)
               for i,file in enumerate(files):
                    print ("Processing image {}/{}".format(i,len(files)))
                    ### Detect face in image
                    pathToImage = os.path.join(path,file)
                    image = cv2.imread(pathToImage)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detector = MTCNN()
                    faces = detector.detect_faces(rgb)
                    boxOfFaces = []
                    for face in faces:
                         ### Check if face detection is true and save that crop face to data
                         print ("confidence:", face["confidence"])
                         boxDetect = face["box"] #box = (x,y-left,top width-height)
                         left, top, right, bottom = abs(boxDetect[0]), abs(boxDetect[1]), abs(boxDetect[0] + boxDetect[2]), abs(boxDetect[1] + boxDetect[3])
                         print (left,top,right,bottom)
                         faceCrop = image[top-padding : bottom+padding, left-padding : right+padding]
                         cv2.namedWindow("crop-face", cv2.WINDOW_GUI_EXPANDED) 
                         cv2.imshow("crop-face", faceCrop)
                         k = cv2.waitKey(5000)
                         if k==ord('y'):
                              nameImage = str(count) + ".jpg"
                              pathToSaveCropFace = os.path.join(pathToFaceFolder,nameImage)
                              print ("Saving ",pathToSaveCropFace)
                              cv2.imwrite(pathToSaveCropFace, faceCrop)
                              boxDetect = (top,right,bottom,left)
                              boxOfFaces.append(boxDetect)
                              count += 1
                              cv2.destroyAllWindows()
                         else:  
                              print ("Not a face")
                              cv2.destroyAllWindows()
                    #encode face to 128d-vector and save to list
                    encodings = face_recognition.face_encodings(rgb, boxOfFaces)
                    for encoding in encodings:
                         knownNames.append(namePerson)
                         knownEncodings.append(encoding)
     ### Save encode with name to pickle file for compare vector (SVM)
     saveDataToPickleFile(knownNames,knownEncodings,pathToSaveEncoding)