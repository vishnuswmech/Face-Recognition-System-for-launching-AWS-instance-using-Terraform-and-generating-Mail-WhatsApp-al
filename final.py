#!/usr/bin/env python
# coding: utf-8

# # Face Recognition System for Provisioning AWS instance using Terraform      scripts and generating Mail & WhatsApp alerts 

# ## Step1: Data collection or Collecting Samples

# In[10]:


import cv2
import numpy as np


# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam 
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'C:/Users/Vishnu/task6-facedetection/vishnu/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# ## Step 2: Model Training

# In[7]:


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = 'C:/Users/Vishnu/task6-facedetection/vishnu/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

vishnu_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
vishnu_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained successfully")


# ## Step3: Face Recognition and performing tasks

# In[8]:


import cv2
import numpy as np
import os
import time

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi




# Open Webcam
cap = cv2.VideoCapture(0)




while True:

    ret, frame = cap.read()
    
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = vishnu_model.predict(face)
        
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident this is Vishnu'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Vishnu", (195, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, "Press Enter to Continue..", (195, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            
            if cv2.waitKey(10) == 13:
             pass
            Confidence90=True
           
         
        else:
            
            cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            if cv2.waitKey(10) == 13:
             break
         
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
    
        if cv2.waitKey(10) == 13:
             break
         
    
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
        
cap.release()
cv2.destroyAllWindows()  

if Confidence90==True:
 os.system("terraform apply --auto-approve")
 time.sleep(6)
 print("Security Group created successfully.......")
 time.sleep(2)
 print("VPC created successfully.......")
 time.sleep(2)
 print("Subnet created successfully.......")
 time.sleep(2)
 print("Route tables created successfully.......")
 time.sleep(2)
 print("Route table Association to Subnet done successfully.......")
 time.sleep(2)
 print("Internet Gateway created successfully.......")
 time.sleep(2)
 print("AWS instance created successfully.......")
 time.sleep(3)
 print("EBS Volume created successfully.......")
 time.sleep(2)
 print("EBS volume attachment to Instance done Successfully......")
 print("Done!!")
        

    


# ## Step1: Data collection or Collecting Samples

# In[11]:


import cv2
import numpy as np


# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam 
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'C:/Users/Vishnu/task6-facedetection/vijay/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# ## Step 2: Model Training

# In[12]:


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = 'C:/Users/Vishnu/task6-facedetection/vijay/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

vijay_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
vijay_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Vijay Model trained sucessfully")


# ## Step3: Face Recognition and performing tasks

# In[17]:


import cv2
import numpy as np
import os
import time
import smtplib, ssl
from email.message import EmailMessage
import pywhatkit as kt
import getpass as gp


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi




# Open Webcam
cap = cv2.VideoCapture(0)




while True:

    ret, frame = cap.read()
    
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = vijay_model.predict(face)
        
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident this is Acto Vijay '
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Vijay", (270, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, "Press Enter to Continue..",(250, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            if cv2.waitKey(10) == 13:
                break
            Confidence90=True
            
         
        else:
            
            cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            if cv2.waitKey(10) == 13:
             break
         
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
    
        if cv2.waitKey(10) == 13:
             break
         
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
        
cap.release()
cv2.destroyAllWindows()


if Confidence90==True:
 

 msg = EmailMessage()
 msg.set_content("Hi Vishnu...This face looks like Actor Vijay")
 msg["Subject"] = "An Email Alert"
 msg["From"] = "vishnuanand97udt@gmail.com"
 msg["To"] = "vishnuanand97udt@gmail.com"

 context=ssl.create_default_context()

 with smtplib.SMTP("smtp.gmail.com", port=587) as smtp:
    smtp.starttls(context=context)
    smtp.login(msg["From"], "fgfhtrhtrthh")
    smtp.send_message(msg)
    print("Mail was Sent Successfully")
 kt.sendwhatmsg_instantly("+918667829231","Hi Vishnu,this face looks like Actor Vijay")
 print("Whatsapp Message was sent successfully")


# ---
