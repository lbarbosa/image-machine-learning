# -*- coding: utf-8 -*-
"""
Pathological image processing program

Equipe de desenvolvimento:
Lourenco Madruga Barbosa


Orientado por:
Fabio Kurt Schineider

"""

#--------------------CONSTANTES------------------#
CONFTHRESHOLD_04 = '0.4'
NMSTHRESHOLD_04  = '0.4'
INPWIDTH         =  512
INPHEIGHT        =  512

ARCHIVE_TYPE = '*.jpg'



DETECT  = 'detect'
CUT_OUT = 'cut_out'


GPU = 'GPU'
CPU = 'CPU'


CLASSES_FILE     = '/home/lbarbosa/darknet_thalita/data/names.names'
MODEL_CONGIG     = '/home/lbarbosa/darknet_thalita/cfg/yolov3.cfg'
MODEL_WEIGHTS    = '/home/lbarbosa/darknet_thalita/backup/yolov3_best.weights'
IMAGE_PATH       = '/home/lbarbosa/darknet_thalita/data/1119_41.jpg'
FOLDER_INIMAGE   = '/home/lbarbosa/darknet_thalita/data/'
FOLDER_OUTIMAGE  = '/home/lbarbosa/darknet_thalita/data/detect/'
PATH_VIDEO_FILE  = ''