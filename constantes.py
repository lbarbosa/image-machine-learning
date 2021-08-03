# -*- coding: utf-8 -*-
"""
Pathological image processing program

Equipe de desenvolvimento:
Lourenco Madruga Barbosa


Orientado por:
Fabio Kurt Schineider

"""

import i18n as i18
import platform
#--------------------CONSTANTES------------------#
#************************************************************
#performBatchImageFile
#************************************************************
DETECT  = 'detect'
CLASSES_FILE        = '/home/lbarbosa/darknet_thalita/data/names.names'
MODEL_CONGIG        = '/home/lbarbosa/darknet_thalita/cfg/yolov3.cfg'
MODEL_WEIGHTS       = '/home/lbarbosa/darknet_thalita/backup/yolov3_best.weights'
IMAGE_PATH          = '/home/lbarbosa/darknet_thalita/data/1119_41.jpg'
FOLDER_INIMAGE      = '/home/lbarbosa/darknet_thalita/data/'
FOLDER_OUTIMAGE     = FOLDER_INIMAGE + 'detect/'
PATH_VIDEO_FILE     = ''
CONFTHRESHOLD_04 = '0.4'
NMSTHRESHOLD_04  = '0.4'
INPWIDTH         =  512
INPHEIGHT        =  512



#************************************************************
#performCutSvsImage
#************************************************************
CUT_OUT = 'cut_out'
PATH_SVS_FILE_IN = 'C:\\Users\\lbarbosa\\Google Drive\\01 - Doutorado\\BASE DE IMAGENS\\BASE DE IMAGENS CRUAS\\LEEDS\\33836\\33836_2'
ZERO             =  0
LEVEL            =  0
OUTPWIDTH        =  1024
OUTPHEIGHT       =  1024
OVERLAP          =  0
JPG              = '.jpg'
SVS              = '.svs'

#************************************************************
#MOUNT IMAGE
#************************************************************
MOUNT_IMAGE = 'mount_image'
PATH_OPENSLIDE = 'E:/image-machine-learning/openslide/bin'
PATH_PDF_FILE_IN = 'C:\\Users\\lbarbosa\\Google Drive\\01 - Doutorado\\BASE DE IMAGENS\\BASE DE IMAGENS BIOATLAS\\\Imagem_1575'



#************************************************************
#Windows/Linux
#************************************************************

SYS_WIN    = 'Windows'
SYS_LINUX  = 'Linux'
LINUX_BAR  = '/'
WIN_BAR    = '\\'
OUT        = '_OUT'


ARC_SKIP_LINE = '\n' 
ARC_MODE_RT   = 'rt'
ARC_TYPE_JPG  = '*.jpg'
ARC_TYPE_SVS  = '*.svs'

GPU = 'GPU'
CPU = 'CPU'

#************************************************************
#Get Operating System
#************************************************************


#************************************************************
#performDatasetPrepare
#************************************************************
DATASET_PREPARE = 'dataset_prepare'
PATH_DS_IMAGES  = 'C:\\Users\\lbarbosa\\Google Drive\\01 - Doutorado\\BASE DE IMAGENS\\BASE DE IMAGENS PREPARADAS\\BASE DE IMAGENS BIOATLAS\\Imagem_1575'
EXPERIMENTO     = 'Exp_BioAtlas_A'
CSV_NAME        = 'Exp_569_BioAtlas_A.csv'
PATH_DS_IMAGES_OUT = PATH_DS_IMAGES + '\\' + EXPERIMENTO + '\\'
PATH_CSV        = PATH_DS_IMAGES + '\\' + CSV_NAME
ARC_TYPE_JPG  = '*.jpg'
DOT_JPG       = '.jpg'
ARC_TYPE_TXT  = '*.txt'
DOT_TXT       = '.txt'

