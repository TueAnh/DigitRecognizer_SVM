# Nguyen Duc Tue Anh
# Digit recognizer using SVM
# -------------------------- 
import datetime
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import SMO_SVM
from skimage.feature import hog
import pandas as pd 
import seaborn as sns
def predict(clf,X):
    return np.dot(clf[:-1],X) + clf[-1]
def predict_class(X, classifiers):
    predictions = np.zeros(len(classifiers))
    for idx, clf in enumerate(classifiers):
        predictions[idx] = predict(clf,X)
    out = np.argmax(predictions[:])
    return out

#load model
classifiers = np.loadtxt("model2.txt")
#predict image
while (True):
    image = cv2.imread("F:\Artificial Intelligence\SVM\Data/3.png")
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    im,thre = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    for i in contours:
        (x,y,w,h) = cv2.boundingRect(i)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        roi = thre[y:y+h,x:x+w]
        roi = np.pad(roi,(28,28),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))    
        x0 =  roi.flatten()
        nbr  = predict_class(x0,classifiers)

        two_d = (np.reshape(x0, (28, 28)) * 255).astype(np.uint8)
        plt.title('Predicted Label: {0}'.format(predict_class(x0,classifiers)))
        plt.imshow(two_d, interpolation='nearest',cmap='gray')
        plt.show()
        
        cv2.putText(image, str(int(nbr)), (x, y),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 153, 255), 3)
        cv2.imshow("image",image)
    cv2.imwrite("image_pand.jpg",image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("Continue? Y/N: ")
    if (input().upper() == "N"):
        break