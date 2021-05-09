# -*- coding: utf-8 -*-
"""
Pathological image processing program

Equipe de desenvolvimento:
Lourenco Madruga Barbosa


Orientado por:
Fabio Kurt Schineider

"""

#--------------------CONSTANTES------------------#
#performBatchImageFile
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


#performCutSvsImage
CUT_OUT = 'cut_out'
PATH_SVS_FILE_IN    = 'C:\\Users\\lbarbosa\\Google Drive\\01 - Doutorado\\BASE DE IMAGENS\\BASE DE IMAGENS JUAN ROSAIS\\131-case15\\img_exp_L_2'
ZERO             =  0
LEVEL            =  0
OUTPWIDTH        =  1024
OUTPHEIGHT       =  1024
OVERLAP          =  0
JPG              = '.jpg'
SVS              = '.svs'



ARC_TYPE_JPG = '*.jpg'
ARC_TYPE_SVS = '*.svs'

GPU = 'GPU'
CPU = 'CPU'





