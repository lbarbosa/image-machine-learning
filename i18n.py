# -*- coding: utf-8 -*-
"""
Programa de processamento de imagens patol0icas

Equipe de desenvolvimento:
Lourenco Madruga Barbosa

Orientado por:
Fabio Kurt Schineider

"""
#************************************************************
# imports
#************************************************************

from pyflakes.messages import RaiseNotImplemented
import locale

#************************************************************
# Classe I18N para internaciolização programa
#************************************************************
class I18N():
    
    def __init__(self):
             
        lang = locale.getdefaultlocale()
             
        if lang[0] == 'en_US': 
            self.languageEnglish()
        elif lang[0] == 'pt_BR' :
            self.languagePortuguese()            
        else:
            if lang[0] == 'en_US':
                RaiseNotImplemented('Unsuported language.')
            elif lang[0] == 'pt_BR':
                RaiseNotImplemented('Idioma não suportado.')
        
        return
    
    def languageEnglish(self):

#ARGUMENT  
        self.TITLE                = 'UTFPR and SHINSHU image processing research using YOLO and Opencv.'
        self.CONFIDENCE_THRESHOLD = 'Confidence threshold.'      
        self.NMSTHRESHOLD         = 'Non-maximum suppression threshold.'
        self.FUNCTION             = 'function that will be executed, detect, cut_out, resize_larger, resize_smaller.'
        self.DEVICE               = 'Device to perform inference on cpu or gpu.'
        self.CLASSES_FILE         = 'Path of class names.'
        self.MODEL_CONGIG         = 'network model configuration'
        self.MODEL_WEIGHTS        = 'trained network model'
        self.IMAGE_PATH           = 'Path to image file.'
        self.FOLDER_INIMAGE       = 'Folder for input images.'
        self.FOLDER_OUTIMAGE      = 'Folder for output images.'
        self.PATH_VIDEO_FILE      = 'Path to video file.'

#DETECT
        self.DETECT_OPITION_START      = 'Detection function selected'
        self.DETECT_OPITION_FINISHED   = 'Detection function finished'   

        self.NET                       = 'Network initialized'                
        self.USING_CPU                 = 'Using CPU device.' 
        self.USING_GPU                 = 'Using GPU device.'           

        self.PATH_TO_UNINFORMED      = 'Path to uninformed image file.'

#CUT OUT 
        self.CUTOUT_OPTION_STARTED  = 'cut out function started '
        self.CUTOUT_OPTION_FINISHED = 'cut out function finished'
        
        self.PATH_SVS_FILE_IN       = 'Path to the SVS input file.'
        self.LEVEL                  = 'Level image SVS'
        self.MESSA_SO_NOT           = 'Not available for OS'
        self.FOLDER_CREATED         = 'Folder & created successfully'
        
#NO FUNCTION
        self.NOFUNC_SELECTED   = 'No function selected'
        
                
#FOOTER
        self.WELCOME_TO     = 'Welcome to the software of Pathological Analysis'
        

    
    
    def languagePortuguese(self):
        
#Argumentos 
        self.TITLE                = 'Pesquisa de processamento de imagem UTFPR e SHINSHU usando YOLO e Opencv.'    
        self.CONFIDENCE_THRESHOLD = 'Limiar de confiança.'
        self.NMSTHRESHOLD         = 'Limite de supressão não máximo.'
        self.FUNCTION             = 'função que será executada, detectar, cut_out, resize_larger, resize_smaller.'
        self.DEVICE               = 'Dispositivo para realizar inferência na cpu ou gpu.'        
        self.CLASSES_FILE         = 'Caminho dos nomes das classes.'
        self.MODEL_CONGIG         = 'configuração do modelo da rede'
        self.MODEL_WEIGHTS        = 'modelo da rede treinado'        
        self.IMAGE_PATH           = 'Caminho para o arquivo de imagem.'
        self.FOLDER_INIMAGE       = 'Pasta para imagens de entrada.' 
        self.FOLDER_OUTIMAGE      = 'Pasta para imagens de saída.' 
        self.PATH_VIDEO_FILE      = 'Caminho para o arquivo de vídeo.'              
                
#DETECT
        self.DETECT_OPITION_START    = 'Função detecção selecionada'
        self.DETECT_OPITION_FINISHED = 'Função detecção finalizada'
 
        self.IMPUT_IMAGE             = 'Input image file ' 
        self.DOSENT_EXIST            = ' doesnt exist'
 
        self.NET                     = 'Rede inicializada'        
        self.USING_CPU               = 'Usando dispositivo de CPU.' 
        self.USING_GPU               = 'Usando dispositivo de GPU.' 
        
        self.PATH_TO_UNINFORMED      = 'Caminho para o arquivo de imagem não informado.'




#CUT OUT 
        self.CUTOUT_OPTION_STARTED  = 'função de corte iniciada'
        self.CUTOUT_OPTION_FINISHED = 'função de corte terminada'

        self.PATH_SVS_FILE_IN       = 'Caminho para o arquivo de entrada SVS.'
        self.LEVEL                  = 'Level image SVS'
        self.MESSA_SO_NOT           = 'Não disponível para o SO'
        self.FOLDER_CREATED         = 'Pasta & criada com sucesso'
        
#NO FUNCTION
        self.NOFUNC_SELECTED   = 'Nenhuma função selecionada'


        
#RODAPE
        self.WELCOME_TO     = 'Bem-vindo ao software de Análise Patológica'
        
            

