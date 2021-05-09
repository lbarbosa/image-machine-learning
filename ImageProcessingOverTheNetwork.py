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
import platform
os.add_dll_directory('E:/image-machine-learning/openslide/bin')
import openslide as open
import glob, os
import constantes
import i18n as i18



def getArgument( i18n ):
    parser = argparse.ArgumentParser(description= i18n.TITLE )
    parser.add_argument('--confThreshold',  default= constantes.CONFTHRESHOLD_04,   help= i18n.CONFIDENCE_THRESHOLD )
    parser.add_argument('--nmsThreshold',   default= constantes.NMSTHRESHOLD_04,    help= i18n.NMSTHRESHOLD)
    parser.add_argument('--funcition',      default= constantes.CUT_OUT,             help= i18n.FUNCTION)
    parser.add_argument('--device',         default= constantes.GPU,                help= i18n.DEVICE)
    parser.add_argument('--classesFile',    default= constantes.CLASSES_FILE,       help= i18n.CLASSES_FILE)
    parser.add_argument('--modelConfig',    default= constantes.MODEL_CONGIG,       help= i18n.MODEL_CONGIG)
    parser.add_argument('--modelWeights',   default= constantes.MODEL_WEIGHTS,      help= i18n.MODEL_WEIGHTS)
    parser.add_argument('--imagePath',      default= constantes.IMAGE_PATH,         help= i18n.IMAGE_PATH)
    parser.add_argument('--folderInImage',  default= constantes.FOLDER_INIMAGE,     help= i18n.FOLDER_INIMAGE)
    parser.add_argument('--folderOutImage', default= constantes.FOLDER_OUTIMAGE,    help= i18n.FOLDER_OUTIMAGE)
    parser.add_argument('--pathVideoFile',  default= constantes.PATH_VIDEO_FILE,    help= i18n.PATH_VIDEO_FILE)
    parser.add_argument('--pathSvsFileIn',  default= constantes.PATH_SVS_FILE_IN,   help= i18n.PATH_SVS_FILE_IN)
#    parser.add_argument('--pathSvsFileOut', default= constantes.PATH_SVS_FILE_OUT,  help= i18n.PATH_SVS_FILE_OUT)
    parser.add_argument('--level',          default= constantes.LEVEL,  help= i18n.LEVEL)
#    parser.add_argument('--level',          default= constantes.PATH_SVS_FILE_OUT,  help= i18n.PATH_SVS_FILE_OUT)
#    parser.add_argument('--pathSvsFileOut', default= constantes.PATH_SVS_FILE_OUT,  help= i18n.PATH_SVS_FILE_OUT)
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

def performGetPathOut (pathIn, count):
    
    title, ext = os.path.splitext(os.path.basename(pathIn))
    if platform.system() == 'Windows':
        path = pathIn.split('\\')
        v_len = len(path) - 2
        path[v_len]
    elif  platform.system() == 'Linux':
        path = pathIn.split('/')
        v_len = len(path) - 2
        path[v_len]
    else:
        print(i18n.MESSA_SO_NOT)
        
    if ext == constantes.SVS :
        ext = constantes.JPG
        
    if (count != 0 ):
        PathOut = constantes.PATH_SVS_FILE_IN + '\\' +path[v_len]+ '_REC'+ '\\'
        if os.path.exists(PathOut) == False:
            os.mkdir(PathOut)
            print(i18n.FOLDER_CREATED.replace('&', PathOut))
            imageOut = PathOut + title + str(count) + ext
        else:
            imageOut = PathOut + title + str(count) + ext
    else:
        imageOut = pathOut + pathOut[len] + title + ext

    return imageOut

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
    hasImage, image = image.read()
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
        PathPut = performGetPathOut( imagePath, 0 )
        cv.imwrite(PathPut, image.astype(np.uint8))
    else:
        vid_writer.write(image.astype(np.uint8))

    cv.imshow("teste", image)   



def performCutSvsImage (pathSvsFileIn,
                        level,
                        outpwidth,
                        outpheight,
                        overlap):
    
    openSlideObj = open.OpenSlide(pathSvsFileIn)    
    W, H = openSlideObj.level_dimensions[level]
    w, h = (outpwidth, outpheight)    
    Sobreposicao_h = overlap
    Sobreposicao_w = overlap
    
    h_dist = h - Sobreposicao_h
    w_dist = w - Sobreposicao_w
        
    div_h = H / h_dist
    div_w = W / w_dist
    
    div_h = int(div_h) 
    div_w = int(div_w) 
    contador = constantes.ZERO
    
    for y in range(div_h):
        for x in range(div_w):
            print("Y", y*h_dist)
            print("X", x*w_dist)
                      
            contador = contador + 1
    
            # Recorta a imagem.
            imagem2 = openSlideObj.read_region((x*w_dist, y*h_dist),level,(int(w), int(h)))
            #imagem2 = openSlideObj.read_region((0, 0),vl_level,(W, H))
            imagem3  = np.array(imagem2,dtype = np.uint8)
    
            #Converte a imagem para RGB padrão Opencv
            r,g,b,a = cv.split(imagem3)
            imagem4 = cv.merge((b,g,r))
    
            #salva imagem no diretorio 
            pathOut = performGetPathOut( pathSvsFileIn, contador )
            cv.imwrite(pathOut, imagem4)

    
#************************************************************
# Program initialization
#************************************************************
if __name__ == "__main__":
    
    #get internationalization
    i18n = i18.I18N()   
    #get arguments
    args = getArgument( i18n ) 
    

#************************************************************
# Initializes detection process
#************************************************************
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
            for pathAndFilename in glob.iglob(os.path.join(args.folderInImage, constantes.ARC_TYPE_JPG )):
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
    
#************************************************************
# Open SVS image, crop and save as JPG 
#************************************************************  
    elif (args.funcition == constantes.CUT_OUT):
       print(i18n.CUTOUT_OPTION_STARTED)
       
       for pathAndFilename in  glob.iglob(os.path.join(args.pathSvsFileIn, constantes.ARC_TYPE_SVS )):
           performCutSvsImage( pathAndFilename, 
                               args.level,
                               constantes.OUTPWIDTH,
                               constantes.OUTPHEIGHT,
                               constantes.OVERLAP)

       print(i18n.CUTOUT_OPTION_FINISHED)
               
    else: 
 
        print(i18n.NOFUNC_SELECTED)