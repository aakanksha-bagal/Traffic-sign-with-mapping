#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import pickle
import os
import keras
import tensorflow as tf
from keras.models import load_model

import random
import string

import subprocess as sp
import re
import time

from tempfile import NamedTemporaryFile
import shutil
import csv

#############################################

filename = 'my.csv'
fields = ['sign', 'latitude', 'longitude', 'radius']
tempfile = NamedTemporaryFile(mode='w', delete=False)


#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(20) 

    
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRANNIED MODEL

# try:
#     pickle_in=open("model_trained.sav","rb")  ## rb = READ BYTE
#     model=pickle.load(pickle_in)
# except EOFError:
#     data=list()

model = load_model('model_d0_3.h5')

def loca():
    wt = 5 # Wait time 
    accuracy = 100

    # while True:
    # time.sleep(wt)
    pshellcomm = ['powershell']
    pshellcomm.append('add-type -assemblyname system.device; '                        '$loc = new-object system.device.location.geocoordinatewatcher;'                        '$loc.start(); '                        'while(($loc.status -ne "Ready") -and ($loc.permission -ne "Denied")) '                        '{start-sleep -milliseconds 100}; '                        '$acc = %d; '                        'while($loc.position.location.horizontalaccuracy -gt $acc) '                        '{start-sleep -milliseconds 100; $acc = [math]::Round($acc*1.5)}; '                        '$loc.position.location.latitude; '                        '$loc.position.location.longitude; '                        '$loc.position.location.horizontalaccuracy; '                        '$loc.stop()' %(accuracy))

    p = sp.Popen(pshellcomm, stdin = sp.PIPE, stdout = sp.PIPE, stderr = sp.STDOUT, text=True)
    (out, err) = p.communicate()
    out = re.split('\n', out)

    lat = float(out[0])
    long = float(out[1])
    radius = int(out[2])
    return lat,long,radius

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'
 
while True:
 
    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # PREDICT IMAGE
    predictions = model.predict(img)
    #classIndex = model.predict_classes(img)
    classIndex = np.argmax(model.predict(img), axis=-1)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    # check for the key pressed
    o = cv2.waitKey(125)
    
    if o == ord('s'):
        
        lat,long,radius = loca() 
        op = open("my.csv", "r")
        dt = csv.DictReader(op)
        up_dt = []
        for row in dt:
            r = {'sign': row['sign'], 'longitude': row['longitude'], 'latitude': row['latitude'], 'radius': row['radius']}
            if row['sign'] != str(getCalssName(classIndex)) and row['longitude'] == str(long) and row['latitude'] == str(lat) and row['radius'] == str(radius) :
                r = {'sign': str(getCalssName(classIndex)), 'longitude': row['longitude'], 'latitude': row['latitude'], 'radius': row['radius']}        
            up_dt.append(r)
        
        op.close()
        
        op = open("my.csv", "w", newline='')
        headers = ['sign', 'longitude', 'latitude', 'radius']
        data = csv.DictWriter(op, delimiter=',', fieldnames=headers)
        data.writerow(dict((heads, heads) for heads in headers))
        data.writerows(up_dt)
        
        op.close()
        
    if o == ord('a'):
        
        lat,long,radius = loca()
        
        op = open("my.csv", "r")
        dt = csv.DictReader(op)
        up_dt = []
        
        for row in dt:
            r = {'sign': row['sign'], 'longitude': row['longitude'], 'latitude': row['latitude'], 'radius': row['radius']}
            up_dt.append(r)
        op.close()
        
        si, lo, la, ra = str(getCalssName(classIndex)), str(long), str(lat), str(radius)
        r = {'sign': si, 'longitude': lo, 'latitude': la, 'radius': ra}
        up_dt.append(r)
        
        op = open("my.csv", "w", newline='')
        headers = ['sign', 'longitude', 'latitude', 'radius']
        data = csv.DictWriter(op, delimiter=',', fieldnames=headers)
        data.writerow(dict((heads, heads) for heads in headers))
        data.writerows(up_dt)
        
        op.close()
    
    if o == 27:
        break
    
    

# close the camera
cap.release()
  
# close all the opened windows
cv2.destroyAllWindows()


# In[ ]:




