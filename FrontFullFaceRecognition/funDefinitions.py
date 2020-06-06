# This module has all the function defections and this is imported into the main module
# Importing necessary libraries
import cv2
import os
import imutils
import numpy as np


# Idea of this function is to make use of HaarCascade intelligence to detect the all human faces in an image
# This take an image, converts into gray scale, uses HaarCascade on the image and finds the faces
# This function will return the each face coordinates and also gray scale image
# Link to download HaarCascade xmls - https://github.com/opencv/opencv/tree/master/data/haarcascades
def faceDetection(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceHaarCascade = cv2.CascadeClassifier('haarCascade/haarcascade_frontalface_default.xml')
    # Adjust scale and minNeig for detecting only faces not random small objects
    facesRectCoordinates = faceHaarCascade.detectMultiScale(grayImg, scaleFactor=1.32, minNeighbors=5)
    return facesRectCoordinates, grayImg


def labelMapping(label):
    targetLabel = os.listdir("data/training")
    return targetLabel.index(label)


# Idea of this function is to make a array of all the gray and gray resized faces from all the training images
# These arrays will be used as the input to the Face Recognition LBPH, Eigen and Fisher algorithms
def trainImagePrep(dirPath):
    facesGray = []
    label = []
    facesResizeGray = []
    for path, subDirNames, fileNames in os.walk(dirPath):
        for imgFileNames in fileNames:
            targetLabel = os.path.basename(path)  # Current director name is assuming as the label
            filePath = os.path.join(path, imgFileNames)
            img = cv2.imread(filePath)
            if img is None:
                continue  # Could be corrupted images or non image files
            facesDectected, grayImg = faceDetection(img)
            if len(facesDectected) != 1:
                continue  # Skipping the images having more than one face - as it will be ineligible for training
            (x, y, w, h) = facesDectected[0]
            faceGrayImg = grayImg[y:y + w, x:x + h]  # Crop face part, will be used in recognition Eigen & Fisher algo
            # Analysed all the faceGrayImg, took the average height and width for resize factor input below
            faceResizeGrayImg = imutils.resize(faceGrayImg, width=130, height=130)
            facesResizeGray.append(faceResizeGrayImg)
            """" # Uncomment the below code only if you want to see all the faces that are used for training
            cv2.imshow("Recognition Algo Input face", faceResizeGrayImg)
            cv2.waitKey(150)
            cv2.destroyAllWindows()
            """
            facesGray.append(faceGrayImg)
            label.append(labelMapping(targetLabel))
        print("Completed reading faces from :", path)
    return facesGray, facesResizeGray, label


# Idea of this function to initialize individual objects to each face detection algorithm and train them
def faceRecognitionAlgos(facesGray, facesResizeGray, indexLabel):
    faceRecognizerLBPHF = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizerEigen = cv2.face.EigenFaceRecognizer_create()
    faceRecognizerFisher = cv2.face.FisherFaceRecognizer_create()
    faceRecognizerLBPHF.train(facesGray, np.array(indexLabel))
    # Eigen and Fisher algos need all the training and testing faces to have same size so we are inputting resize images
    faceRecognizerEigen.train(facesResizeGray, np.array(indexLabel))
    faceRecognizerFisher.train(facesResizeGray, np.array(indexLabel))
    return faceRecognizerLBPHF, faceRecognizerEigen, faceRecognizerFisher


# This function is to draw a rectangle around the detected face
def rectangle(img, faceCoordinates, borderC, thickness):
    (x, y, w, h) = faceCoordinates
    cv2.rectangle(img, (x, y), (x + w, y + h), borderC, thickness=thickness)

