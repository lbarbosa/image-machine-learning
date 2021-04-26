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
    parser.add_argument('--classesFile',    default= constantes.CLASSESFILE,        help= i18n.CLASSESFILE)
    parser.add_argument('--imagePath',      default= constantes.IMAGEPATH,          help= i18n.IMAGEPATH)
    parser.add_argument('--folderInImage',  default= constantes.FOLDERINIMAGE,      help= i18n.FOLDERINIMAGE)
    parser.add_argument('--folderOutImage', default= constantes.FOLDEROUTIMAGE,     help= i18n.FOLDEROUTIMAGE)
    parser.add_argument('--pathVideoFile',  default= constantes.PATHVIDEOFILE,      help= i18n.PATHVIDEOFILE)
    args = parser.parse_args()
    return args


#Process batch image file
def performBatchImageFile (imagePath,
                           imagePathPut,
                           thresh ,
                           net,
                           meta, 
                           hier_thresh, 
                           nms):

    if imagePath == '':
        print("erro")


if __name__ == "__main__":
    
    #get internationalization
    i18n = i18.I18N() 
    
    #get arguments
    args = getArgument( i18n ) 
    
    if (args.funcition == constantes.DETECT):
        print( i18n.DETECT_OPITION_START )
       
        if(args.folderInImage == ' '):    
            for pathAndFilename in glob.iglob(os.path.join(args.image_folder, constantes.ARCHIVE_TYPE )):
                performBatchImageFile ( pathAndFilename,
                                        args.imageFolderOut,
                                                                      
                                    
                                     )
            
        
        print(i18n.DETECT_OPITION_FINISHED)
        
    elif (args.funcition == constantes.CUT_OUT):
       print(i18n.CUTOUT_OPTION_STARTED)
       
       print(i18n.CUTOUT_OPTION_FINISHED)
               
    else: 
 
        print(i18n.NOFUNC_SELECTED)
        