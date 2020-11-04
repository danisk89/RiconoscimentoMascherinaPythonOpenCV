from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import time
import os
import cv2
import numpy as np
from time import sleep
import os
import _thread

def speakTTS(faceNoMask,a,b):
        if faceNoMask == 0 :
            if a==0 :
                os.system('echo "" | festival --tts')
            else:
                if a==1 :
                    os.system('echo "Pass the check" | festival --tts')
                else:
                    os.system('echo "All the '+str(a)+' persons pass the check" | festival --tts')
        else:
            if faceNoMask == 1 :
                os.system('echo "There is '+str(faceNoMask)+' person without mask protection" | festival --tts')
            else:
                os.system('echo "There are '+str(faceNoMask)+' persons without mask protection" | festival --tts')

def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from ios.systet
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
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

                        # ensure the bounding boxes fall within the dimensions >
                        # the frame
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        # extract the face ROI, convert it from BGR to RGB chan>
                        # ordering, resize it to 224x224, and preprocess it
                        face = frame[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)

                        # add the face and bounding boxes to their respective
                        # lists
                        faces.append(face)
                        locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('/mnt/hgfs/Shared/video.mp4')
#cap = cv2.VideoCapture('http://192.168.1.197:4747/video')
#cap = cv2.VideoCapture('/home/pi/Face-Mask-Detection/images/video2.mp4')

# Check if camera opened successfully
#if (cap.isOpened()== False): 
#  print("Error opening video stream or file")

# Read until video is completed
#while(cap.isOpened()):
  # Capture frame-by-frame
#  ret, frame = cap.read()
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
index = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  frame = frame.array
  ret = True
  if ret == True:
#    frame = cv2.resize(frame, (281, 500))
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    faceNoMask = 0
    faceMask= 0
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
#        label = "Mask" if mask > withoutMask else "No Mask"
        label = "Mask" if mask > 0.85 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        if label=="Mask":
            faceMask=faceMask+1
        else:
             faceNoMask=faceNoMask+1
        cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if index==10 : 
       _thread.start_new_thread(speakTTS, (faceNoMask,faceMask,1))
       index = 0
    index = index + 1
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    rawCapture.truncate(0)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
