# This is the main module which calls the face detection on training data and face recognition on the test data
# Importing necessary libraries
from typing import List, Tuple

import funDefinitions as fd
import cv2
import imutils
import os

targetLabel = os.listdir("data/training")
# Getting all the faces and their target labels from training data set
# Saving the training data into yml files. From the 2nd run, if there are no modification in the training data set
# Comment this part and use the next block to read the intelligence from yml files
# instead of recreating same from the scratch
facesGray, facesResizeGray, label = fd.trainImagePrep(dirPath="data/training")
faceRecognizerLBPHF, faceRecognizerEigen, faceRecognizerFisher = fd.faceRecognitionAlgos(facesGray,
                                                                                         facesResizeGray, label)
faceRecognizerLBPHF.write('trainingDataLBPHF.yml')
faceRecognizerEigen.write('trainingDataEigen.yml')
faceRecognizerFisher.write('trainingDataFisher.yml')


"""# If this code is uncommented make sure to comment the above block
# Initializing and reading face recognition intelligence from the yml
faceRecognizerLBPHF = cv2.face.LBPHFaceRecognizer_create()
faceRecognizerLBPHF.read('trainingDataLBPHF.yml')
faceRecognizerEigen = cv2.face.EigenFaceRecognizer_create()
faceRecognizerEigen.read('trainingDataEigen.yml')
faceRecognizerFisher = cv2.face.FisherFaceRecognizer_create()
faceRecognizerFisher.read('trainingDataFisher.yml')
"""


# This module takes images  stored in TestImages and performs face recognition
testImg = cv2.imread('data/testing/Cocktail_1.jpg')  # test_img path
facesDetected, grayImg = fd.faceDetection(testImg)
print("faces detected in the testing image: ", facesDetected)

for faceDectected in facesDetected:
    (x, y, w, h) = faceDectected
    faceGrayImg = grayImg[y:y+h, x:x+h]
    faceResizeGrayImg = imutils.resize(faceGrayImg, width=130, height=130)
    predLBPHFLabel, confidenceLBPHF = faceRecognizerLBPHF.predict(faceGrayImg)
    predEigenLabel, confidenceEigen = faceRecognizerEigen.predict(faceResizeGrayImg)
    predFisherLabel, confidenceFisher = faceRecognizerFisher.predict(faceResizeGrayImg)
    print("LBPHF : confidence ", confidenceLBPHF, " predicted as ", predLBPHFLabel)
    print("Eigen : confidence ", confidenceEigen, " predicted as ", predEigenLabel)
    print("Fisher : confidence ", confidenceFisher, " predicted as ", predFisherLabel,"\n")
    rectBorderColor = (117,60,55)
    fd.rectangle(testImg, faceDectected,rectBorderColor,thickness=1)
    # Confidence is the error score, less confidence score implies better prediction
    if confidenceLBPHF < 150:
        fillCLBPHF = (255, 0, 0)  # Blue for LBPHF predictions
        cv2.putText(testImg,targetLabel[predLBPHFLabel] ,
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.1, fillCLBPHF, 1)
    if confidenceEigen < 150:
        fillCEigen = (0, 255, 0)  # Green for Eigen predictions
        cv2.putText(testImg, targetLabel[predEigenLabel],
                    (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.1, fillCEigen, 1)
    if confidenceFisher < 150:
        fillCFisher = (0, 0, 255)  # Red for Fisher predictions
        cv2.putText(testImg, targetLabel[predFisherLabel],
                    (x, y+10), cv2.FONT_HERSHEY_PLAIN, 1.1, fillCFisher, 1)

cv2.imshow("face detection tutorial", testImg)
cv2.waitKey(0)  # Waits indefinitely until a key is pressed
cv2.destroyAllWindows()


