# -*- coding: utf-8 -*-
"""
Pathological image processing program

Equipe de desenvolvimento:
Lourenco Madruga Barbosa

Orientado por:
Fabio Kurt Schineider

"""
#************************************************************
# imports
#************************************************************
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import glob, os
import constantes
import i18n as i18


def getArgument( i18n ):
    parser = argparse.ArgumentParser(description= i18n.TITLE )
    parser.add_argument('--confThreshold',  default= constantes.CONFTHRESHOLD_04,   help= i18n.CONFIDENCE_THRESHOLD )
    parser.add_argument('--nmsThreshold',   default= constantes.NMSTHRESHOLD_04,    help= i18n.NMSTHRESHOLD)
    parser.add_argument('--funcition',      default= constantes.DETECT,             help= i18n.FUNCTION)
    parser.add_argument('--device',         default= constantes.GPU,                help= i18n.DEVICE)
    parser.add_argument('--classesFile',    default= constantes.CLASSES_FILE,       help= i18n.CLASSES_FILE)
    parser.add_argument('--modelConfig',    default= constantes.MODEL_CONGIG,       help= i18n.MODEL_CONGIG)
    parser.add_argument('--modelWeights',   default= constantes.MODEL_WEIGHTS,      help= i18n.MODEL_WEIGHTS)
    parser.add_argument('--imagePath',      default= constantes.IMAGE_PATH,         help= i18n.IMAGE_PATH)
    parser.add_argument('--folderInImage',  default= constantes.FOLDER_INIMAGE,     help= i18n.FOLDER_INIMAGE)
    parser.add_argument('--folderOutImage', default= constantes.FOLDER_OUTIMAGE,    help= i18n.FOLDER_OUTIMAGE)
    parser.add_argument('--pathVideoFile',  default= constantes.PATH_VIDEO_FILE,    help= i18n.PATH_VIDEO_FILE)
    args = parser.parse_args()
    return args


def performGetRede(modelConfiguration, modelWeights):
   try:
       net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) 
       print(i18n.NET)
       return net
   except:
       print("An exception occurred")

    
def performGetDevice (net):

    if(args.device == constantes.CPU ):
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print(i18n.USING_CPU)
    elif(args.device == constantes.GPU ):
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print(i18n.USING_GPU)
    return net

def performGetClassesNames ( classesFile ):
    
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def getOutputsNames(net):

    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(image, classes,  classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 2)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(image, classes, outs, confThreshold, nmsThreshold):
    imageHeight = image.shape[0]
    imageWidth  = image.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > float(confThreshold):
                center_x = int(detection[0] * imageWidth)
                center_y = int(detection[1] * imageHeight)
                width = int(detection[2] * imageWidth)
                height = int(detection[3] * imageHeight)
                left = int(center_x - width / 2)
                top  = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, float(confThreshold), float(nmsThreshold))
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(image, classes, classIds[i], confidences[i], left, top, left + width, top + height) 
        
#Process batch image file
def performBatchImageFile (imagePath,
                           imagePathPut,
                           confThreshold ,
                           classes,
                           net,
                           inpWidth,
                           inpHeight,
                           nmsThreshold ):

    # Open the image file for processing
    if not os.path.isfile(imagePath):
        print(i18n.IMPUT_IMAGE , imagePath, i18n.DOSENT_EXIST )
        sys.exit(1)
   
    # get image
    image = cv.VideoCapture(imagePath)
    hasFrame, image = image.read()
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    postprocess(image, classes, outs, confThreshold, nmsThreshold)
  
     # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (imagePath):
        title, ext = os.path.splitext(os.path.basename(imagePath))
        PathPut = imagePathPut + title + ext
        cv.imwrite(PathPut, image.astype(np.uint8))
    else:
        vid_writer.write(image.astype(np.uint8))

    cv.imshow("teste", image)   
    
if __name__ == "__main__":
    
    #get internationalization
    i18n = i18.I18N() 
    
    #get arguments
    args = getArgument( i18n ) 
    
    #Initializes detection process
    if (args.funcition == constantes.DETECT):
        print( i18n.DETECT_OPITION_START )

        #Return classes name
        classesNames = performGetClassesNames ( args.classesFile )
        
        #Return network
        net = performGetRede(constantes.MODEL_CONGIG, 
                             constantes.MODEL_WEIGHTS)
        
        #Selects the CPU or GPU processing device
        performGetDevice(net)
        
        #processes the batch of images reported for the function
        if(args.folderInImage != ' '):              
            for pathAndFilename in glob.iglob(os.path.join(args.folderInImage, constantes.ARCHIVE_TYPE )):
                performBatchImageFile ( pathAndFilename,
                                        args.folderOutImage,
                                        args.confThreshold,
                                        classesNames,
                                        net,
                                        constantes.INPWIDTH,
                                        constantes.INPHEIGHT,
                                        args.nmsThreshold )
             
        else:
            print( i18n.PATH_TO_UNINFORMED )
            sys.exit(1)
        
        print(i18n.DETECT_OPITION_FINISHED)
        
    elif (args.funcition == constantes.CUT_OUT):
       print(i18n.CUTOUT_OPTION_STARTED)
       
       print(i18n.CUTOUT_OPTION_FINISHED)
               
    else: 
 
        print(i18n.NOFUNC_SELECTED)
        