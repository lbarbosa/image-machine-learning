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
        self.CLASSESFILE          = 'Path of class names.'
        self.IMAGEPATH            = 'Path to image file.'
        self.FOLDERINIMAGE        = 'Folder for input images.'
        self.FOLDEROUTIMAGE       = 'Folder for output images.'
        self.PATHVIDEOFILE        = 'Path to video file.'

#DETECT
        self.DETECT_OPITION_START      = 'Detection function selected'
        self.DETECT_OPITION_FINISHED   = 'Detection function finished'   
        
        
        
#CUT OUT 
        self.CUTOUT_OPTION_STARTED  = 'cut out function started '
        self.CUTOUT_OPTION_FINISHED = 'cut out function finished'
        
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
        self.CLASSESFILE          = 'Caminho dos nomes das classes.'
        self.IMAGEPATH            = 'Caminho para o arquivo de imagem.'
        self.FOLDERINIMAGE        = 'Pasta para imagens de entrada.' 
        self.FOLDEROUTIMAGE       = 'Pasta para imagens de saída.' 
        self.PATHVIDEOFILE        = 'Caminho para o arquivo de vídeo.'              
                
#DETECT
        self.DETECT_OPITION_START    = 'Função detecção selecionada'
        self.DETECT_OPITION_FINISHED = 'Função detecção finalizada'




#CUT OUT 
        self.CUTOUT_OPTION_STARTED  = 'função de corte iniciada'
        self.CUTOUT_OPTION_FINISHED = 'função de corte terminada'


#NO FUNCTION
        self.NOFUNC_SELECTED   = 'Nenhuma função selecionada'


        
#RODAPE
        self.WELCOME_TO     = 'Bem-vindo ao software de Análise Patológica'
        
            

