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
import constantes
import i18n as i18
import cv2 as cv
import csv
import argparse
import sys
import numpy as np
import os.path
import platform
import shutil
from numpy.lib.function_base import append
import glob, os
os.add_dll_directory(constantes.PATH_OPENSLIDE)
import openslide as opensl

def getArgument( i18n ):
    parser = argparse.ArgumentParser(description= i18n.TITLE )
    parser.add_argument('--confThreshold',  default= constantes.CONFTHRESHOLD_04,   help= i18n.CONFIDENCE_THRESHOLD )
    parser.add_argument('--nmsThreshold',   default= constantes.NMSTHRESHOLD_04,    help= i18n.NMSTHRESHOLD)
    parser.add_argument('--function',       default= constantes.DATASET_PREPARE,    help= i18n.FUNCTION)
    parser.add_argument('--device',         default= constantes.GPU,                help= i18n.DEVICE)
    parser.add_argument('--classesFile',    default= constantes.CLASSES_FILE,       help= i18n.CLASSES_FILE)
    parser.add_argument('--modelConfig',    default= constantes.MODEL_CONGIG,       help= i18n.MODEL_CONGIG)
    parser.add_argument('--modelWeights',   default= constantes.MODEL_WEIGHTS,      help= i18n.MODEL_WEIGHTS)
    parser.add_argument('--imagePath',      default= constantes.IMAGE_PATH,         help= i18n.IMAGE_PATH)
    parser.add_argument('--folderInImage',  default= constantes.FOLDER_INIMAGE,     help= i18n.FOLDER_INIMAGE)
    parser.add_argument('--folderOutImage', default= constantes.FOLDER_OUTIMAGE,    help= i18n.FOLDER_OUTIMAGE)
    parser.add_argument('--pathVideoFile',  default= constantes.PATH_VIDEO_FILE,    help= i18n.PATH_VIDEO_FILE)
    parser.add_argument('--pathSvsFileIn',  default= constantes.PATH_SVS_FILE_IN,   help= i18n.PATH_SVS_FILE_IN)
    parser.add_argument('--level',          default= constantes.LEVEL,              help= i18n.LEVEL)
    parser.add_argument('--pathPdfFileIn',  default= constantes.PATH_PDF_FILE_IN,   help= i18n.PATH_PDF_FILE_IN)
    parser.add_argument('--pathDsImages',   default=constantes.PATH_DS_IMAGES,       help=i18n.PATH_PDF_FILE_IN)

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
    
    PathOut  = ''
    imageOut = ''
    title, ext = os.path.splitext(os.path.basename(pathIn))
    if platform.system() == constantes.SYS_WIN:
       path = pathIn.split(constantes.WIN_BAR)
       v_bar = constantes.WIN_BAR
       v_len = len(path) - 2
       path[v_len]
    elif  platform.system() == constantes.SYS_LINUX:
       path = pathIn.split(constantes.LINUX_BAR)
       v_bar = constantes.LINUX_BAR
       v_len = len(path) - 2
       path[v_len]
    else:
       print(i18n.MESSA_SO_NOT)   
    if ext == constantes.SVS :
        ext = constantes.JPG
        
    if (count != 0 ):
        PathOut = os.path.dirname(pathIn) + v_bar + path[v_len]+ constantes.OUT + v_bar
        if os.path.exists(PathOut) == False:
            os.mkdir(PathOut)
            print(i18n.FOLDER_CREATED.replace('&', PathOut))
            imageOut = PathOut + title + str(count) + ext
        else:
            imageOut = PathOut + title + str(count) + ext
    else:
        if ext != '':
            imageOut = PathOut + PathOut[len] + title + ext
        else:
            PathOut = pathIn + v_bar + title + constantes.OUT + v_bar
            if os.path.exists(PathOut) == False:
                os.mkdir(PathOut)
                print(i18n.FOLDER_CREATED.replace('&', PathOut))

    return (PathOut, imageOut)

def performGetfileProperties(imagePath):
    retorno = []
    count = 0
    files = os.listdir(imagePath)
    listDir = [i for i in files if i.endswith('.jpg')]
    countImage = len(listDir)
    pathImage =  imagePath + constantes.WIN_BAR + listDir[0] 
    imageName = listDir[0].replace('0.jpg', '' )
    imagem  = cv.imread(pathImage)
    width  = imagem.shape[1]
    #retorno.append(width)
    height = imagem.shape[0]
    #retorno.append(height)
    for image in range(countImage):
        print(pathImage.replace('_0.', '_'+ str(image) + '.'))
        imagem  = cv.imread(pathImage.replace('_0.', '_'+ str(image) + '.'))
        w = imagem.shape[1]
        h = imagem.shape[0]    
        if ( width == w ) and ( height == h ):
            count = count + 1
        else:
            count = count + 1
            break
        
    #retorno.append(count)
    div_h = countImage / count
    #retorno.append(int(div_h))
    #return retorno
    return ( imageName, width, height, count, int(div_h)  )
        
def performGetClassesNames ( classesFile ):
    
    classes = None
    with open(classesFile, constantes.ARC_MODE_RT) as f:
        classes = f.read().rstrip(constantes.ARC_SKIP_LINE).split(constantes.ARC_SKIP_LINE)
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
                        pathImage,
                        level,
                        outpwidth,
                        outpheight,
                        overlap):
    
    openSlideObj = opensl.OpenSlide(pathImage)
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
                    
            contador = contador + 1
   
            # Recorta a imagem.
            imagem2 = openSlideObj.read_region((x*w_dist, y*h_dist),level,(int(w), int(h)))
            #imagem2 = openSlideObj.read_region((0, 0),vl_level,(W, H))
            imagem3  = np.array(imagem2,dtype = np.uint8)
    
            #Convert the image to standard RGB Opencv
            r,g,b,a = cv.split(imagem3)
            imagem4 = cv.merge((b,g,r))
    
            pathOut = performGetPathOut( pathImage, contador )
            #save image to directory 
            cv.imwrite(pathOut[1], imagem4)
            print(pathOut[1])

def performMountImage (imagePath):

    properties = performGetfileProperties(imagePath)
    
    # from PIL import Image, ImageFilter
    w_tile, h_tile = (properties[1], properties[2]) # typical
    multiples = 2
    image_factor_w = 2 * multiples # if = 2 there will be 8 width tiles, if 3 there will be 12 width tiles
    image_factor_h = 2 * multiples # if = 2 there will be 6 high tiles, if 3 there will be 9
    # This is teh size of the output new bigger TILES for latter Deep Learning processing
    w_imageOut = w_tile * image_factor_w
    h_imageOut = h_tile * image_factor_h
    print(w_imageOut, h_imageOut )
    # This is the Original size of the Whole Slide Image (example = 33000 x 24000)
    #w_Orig_Entire_image, h_Orig_Entire_image = ()
    #div_h = h_Orig_Entire_image / h_tile
    #div_w = w_Orig_Entire_image / w_tile
    # This is the information on the number of X and Y tiles
    div_w = properties[3] #81 #113 # confirm and edit HERE after downloading images
    div_h = properties[4] #46 #92 # confirm and edit HERE after downloading images
     
    #717 76 66
    # 867 84 38
     
    PathPut = performGetPathOut( imagePath, 0 )
    # assuming numbered images prepare to read the first one
    i_th_image = 0  # the very first index for the input tiles
    big_tile_number = 0
    for y in range(0, div_h, image_factor_h):
        for x in range(0, div_w, image_factor_w):
            # Need for controlling the new big tiles formation
            ###output_image_4_3_factor = np.zeros((h_imageOut, w_imageOut,4), dtype=np.uint8)
            # this might be 8x6 or 12x9
            #y_internal_value = 8  # might be 12
            #x_internal_value = 6  # might be 9
            #image_factor_h2 = image_factor_h
            if(div_h - y < image_factor_h):
                image_factor_h2 = div_h - y
            else:
                image_factor_h2 = image_factor_h
            print(y, image_factor_h2)
      
             
            if(div_w - x < image_factor_w):
                image_factor_w2 = div_w - x
            else:
                image_factor_w2 = image_factor_w
            print(x, image_factor_w2)
     
            for y1 in range(image_factor_h2):
                for x1 in range(image_factor_w2):
                    i_th_image = (y+y1)*div_w + x+x1
                    input_name_ith = imagePath + constantes.WIN_BAR + properties[0] + str(i_th_image) + constantes.JPG
                    print(input_name_ith)
                     
                    #print(i_th_image)
                    img = cv.imread(input_name_ith)
                    # Converte a imagem para RGB opencv
                    #r, g, b = cv2.split(img)
                    #img_in_rgb = cv2.merge((b, g, r))
                    img_in_rgb = img
                    if(x1==0):
                        matrix_out_v = img_in_rgb
                    else:
                        matrix_out_v = np.hstack((matrix_out_v,img_in_rgb))
                    #output_image_4_3_factor() = img_in_rgb
                     
                if(y1==0):
                        matrix_out_h = matrix_out_v
                else:
                        matrix_out_h = np.vstack((matrix_out_h, matrix_out_v))
            big_tile_number = big_tile_number + 1
            print('preparing new image', big_tile_number)
            output_name_ith = PathPut[0] + properties[0] + str(big_tile_number) + constantes.JPG
            print(output_name_ith)
            output_image_4_3_factor = matrix_out_h
            cv.imwrite(output_name_ith, output_image_4_3_factor)        
    
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
    if (args.function == constantes.DETECT):
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
                print(pathAndFilename)
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
    elif args.function == constantes.CUT_OUT:
       print(i18n.CUTOUT_OPTION_STARTED)
       
       for pathImage in glob.iglob(os.path.join(args.pathSvsFileIn, constantes.ARC_TYPE_SVS )):
           performCutSvsImage( args.pathSvsFileIn,
                               pathImage, 
                               args.level,
                               constantes.OUTPWIDTH,
                               constantes.OUTPHEIGHT,
                               constantes.OVERLAP)

       print(i18n.CUTOUT_OPTION_FINISHED)
       
#************************************************************
# Open small JPG image and mount and save in large JPG
#************************************************************  
    elif args.function == constantes.MOUNT_IMAGE:
        print(i18n.MOUNT_IMAGE_OPTION_STARTED)
        performMountImage (args.pathPdfFileIn
                           )

        print(i18n.MOUNT_IMAGE_OPTION_FINISHED)

    elif args.function == constantes.DATASET_PREPARE:
        print(i18n.DATASET_PREPARE_OPTION_STARTED)
        if os.path.exists(constantes.PATH_DS_IMAGES_OUT) == False:
            os.mkdir(constantes.PATH_DS_IMAGES_OUT)
            print("arquivo criado")
        else:
            print("arquivo ja existe")
        arq_csv = csv.writer(open(constantes.PATH_CSV, "w", newline='', encoding='utf-8'))
        arq_csv.writerow(['Imagen','Categoria do objeto','Eixo x','Eixo y','width','height'])
        for pathImage in glob.iglob(os.path.join(args.pathDsImages, constantes.ARC_TYPE_JPG)):
            title, ext = os.path.splitext(os.path.basename(pathImage))
            print(pathImage)
            pathImage = pathImage.replace(constantes.DOT_JPG, constantes.DOT_TXT)
            print(pathImage)
            if os.path.exists(pathImage):
                with open(pathImage, constantes.ARC_MODE_RT) as arquivo:
                    conteudo = arquivo.read()
                    conteudo = conteudo.split("\n")
                    for linha in conteudo:
                        print(linha)
                        linha = linha.split()
                        arq_csv.writerow([title, linha[0], linha[1], linha[2], linha[3], linha[4]])
                print(pathImage)
                arquivo.close()

                jpg_source =  args.pathDsImages + '\\' + title + constantes.DOT_JPG
                jpg_destination = constantes.PATH_DS_IMAGES_OUT + title + constantes.DOT_JPG
                shutil.copy2(jpg_source, jpg_destination)
                txt_source =  args.pathDsImages + '\\' + title + constantes.DOT_TXT
                txt_destination = constantes.PATH_DS_IMAGES_OUT + title + constantes.DOT_TXT
                shutil.copy2(txt_source, txt_destination)
            else:
                arq_csv.writerow([title,'','','','',''])
                print("The file does not exist")
        print(i18n.DATASET_PREPARE_OPTION_FINISHED)
    else:
        print(i18n.NOFUNC_SELECTED)