import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from IPython.display import Image
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import cv2
import easydict
import matplotlib.pyplot as plt
import io
import picamera
import smtplib
import lcddriver
import time
import datetime
import subprocess

from wordpress_xmlrpc import Client
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc import WordPressPost
import yaml

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

def helloworld(self, params, packet):
    print ('Recieved message from AWS IoT Core')
    print ('Topic: ' + packet.topic)
    print ("Payload: ", (packet.payload))
    
# For certificate based connection
myMQTTClient = AWSIoTMQTTClient("BenClientID")
# For TLS mutual authentication
myMQTTClient.configureEndpoint("a26rjji1og3jji-ats.iot.us-east-2.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
myMQTTClient.configureCredentials("/home/pi/Python/face_detection/root-CA.pem",
                                  "/home/pi/Python/face_detection/private.pem.key",
                                  "/home/pi/Python/face_detection/certificate.pem.crt") #Set path for Root CA and unique device credentials (use the private key and certificate retrieved from the logs in Step 1)
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)

print('Connecting to AWS IoT Core...')
myMQTTClient.connect()

args = easydict.EasyDict({
    "face_deploy": 'utility/face_detector/deploy.prototxt',
    "face_caffe": 'utility/face_detector/res10_300x300_ssd_iter_140000.caffemodel',
    "input_image": 'utility/katie_mask.jpg',
    "confidence": 0.5
})

#set the LCD display, clear previous text
display = lcddriver.lcd()
display.lcd_clear()

#Set the display default text
display.lcd_display_string("  Mask Monitor  ", 1)
display.lcd_display_string("     Module      ", 2)

#load the model from disk
print("Loading model...")
model = load_model("/home/pi/Python/face_detection/mask_detector.model")

#get a summary of the model
#model.summary()

#load the face detector
prototxtPath = args["face_deploy"]
weightsPath = args["face_caffe"]
net = cv2.dnn.readNet(prototxtPath, weightsPath)

numMasks = 0
numNoMasks = 0
totalFaces = 0
count = 0

while (count < 20):
    display.lcd_display_string("  Mask Monitor  ", 1)
    display.lcd_display_string("     Module      ", 2)
    print("[INFO] capturing image from Pi camera...")
    #use the picamera to capture a picture
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.capture('utility/test_mask.jpg')

    #load the image
    test_img = mpimg.imread('utility/test_mask.jpg')
    #orig = test_img.copy()
    (h,w) = test_img.shape[:2]

    blob = cv2.dnn.blobFromImage(test_img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = test_img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            if mask > withoutMask:
                numMasks += 1
                text = "Thank you for wearing a mask"
                os.system('espeak -s150 "'+text+'" 2>/dev/null')
                display.lcd_display_string(" Thank you for  ", 1)
                display.lcd_display_string(" wearing a mask ", 2)
            else:
                numNoMasks += 1
                text = "Please put a mask on!"
                os.system('espeak -s150 "'+text+'" 2>/dev/null')
                display.lcd_display_string("  Please wear   ", 1)
                display.lcd_display_string("    a mask!!    ", 2)
                
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(test_img, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.rectangle(test_img, (startX, startY), (endX, endY), color, 2)
            
            totalFaces += 1;
    count += 1
    time.sleep(2)

print("Number of people with masks: {}".format(numMasks))
print("Number of people without masks: {}".format(numNoMasks))
print("Number of faces detected: {}".format(totalFaces))

#blog = Client('http://localhost/xmlrpc.php', 'bek_capstone', 'Mnidf719L!)')

x = datetime.datetime.now()

print ("Publishing message from RPi")
myMQTTClient.publish(
    topic = "RealTimeDataTransfer/MaskDetections",
    QoS = 0,
    payload = '{"location": "DUC", "masks": '+str(numMasks)+', "noMasks": '+str(numNoMasks)+'}')

#post = WordPressPost()
#post.title = 'Mask Surveillence Results for ' + (x.strftime("%x"))
#post.content = 'Number of people with masks: {}'.format(numMasks)
#post.id = blog.call(posts.NewPost(post))
#post.post_status = 'publish'
#blog.call(posts.EditPost(post.id, post))

#plt.figure(figsize=(40,20))
#imgplot = plt.imshow(test_img)
#plt.show()
