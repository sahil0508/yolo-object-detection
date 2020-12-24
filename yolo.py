import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320

classesFile = 'classNames.txt'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().strip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov3.cfg.txt'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
confidence_threshold = 0.6
nms_threshold = 0.4

def findObjects(outputs,img):
    ht,wt,ct = img.shape
    bbox = []

    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confidence_threshold:
                w,h = int(detection[2]*wt) , int(detection[3] * ht)
                x,y = int((detection[0]*wt) - w/2),int((detection[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confidence_threshold,nms_threshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]]} {int(confs[i]*100)}%',
                 (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
while True:
    success,img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(layerNames)
    # print(net.getUnconnectedOutLayers())

    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape,outputs[1].shape,outputs[2].shape)
    findObjects(outputs,img)
    cv2.imshow('image',img)
    cv2.waitKey(1)