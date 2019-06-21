import requests
import json
import numpy as np
import cv2
import sys
import base64
from time import sleep
import utils
from pathlib import Path

class operations:

    tenantid = '103042019'
    tenantkey = 'sf3ai8BRsHUllijXmkWDCheXGgdr9cab'
    utilidad = utils.kycfaceid
    
    call = False

    def Recognize(self, frame):
        cv2.imwrite('recognize.png',frame)
   
        myPath = Path().absolute()

        reconoce = self.utilidad.recognize( str(myPath) + "/recognize.png", self.tenantid)

        #print(reconoce)

        return json.dumps(reconoce) 


    def addUser(self,name, lastname, details, idIn):
    
        add = self.utilidad.userCreate(name,lastname,details,idIn,self.tenantid)
        #print(add)
        return json.dumps(add)

    def addFace(self, frame, userId):
        
        cv2.imwrite('addface.png',frame)
        myPath = Path().absolute()

        face = self.utilidad.userAddFace(str(myPath) + "/addface.png", userId,self.tenantid)
        #print(face)
        return json.dumps(face)
