import requests
import json
import numpy as np
import cv2
import sys
import base64
from time import sleep

class operations:

    tenantid = '103042019'
    tenantkey = 'sf3ai8BRsHUllijXmkWDCheXGgdr9cab'
    username = 'test'
    password = 'test'
    url ='https://kycface.mooo.com/'
    urlLocal ='http://172.22.4.41/'

    call = False

    def getToken(self):

        headers = {'Content-Type': 'application/json; charset=utf-8', 'Accept': 'application/json', 'tenantid': self.tenantid, 'tenantkey': self.tenantkey}
        dataToken = json.dumps({'username': self.username, 'password': self.password})

        respToken = requests.post( self.urlLocal + 'login',headers = headers,data = dataToken)

        if respToken.status_code != 200:
            print(respToken.status_code)

        token = respToken.json()['access_token']

        return token


    def Recognize(self, frame, token):
        cv2.imwrite('recognize.png',frame)
        baseUri =""

        with open("recognize.png", "rb") as imageFile:
            encoded = base64.b64encode(imageFile.read()).decode()
            baseUri = 'data:image/png;base64,{}'.format(encoded)
        
        
        headers = {"Authorization": 'Bearer ' + str(token),'tenantid': self.tenantid, 'tenantkey': self.tenantkey}
        dataReco = {"image": baseUri}
        
        respReco = requests.post(self.urlLocal + 'kycfaceid/v1/image/recognize',headers = headers,data = dataReco)
                
        return json.loads(respReco.text)


    def addUser(self,name, lastname, details, idIn, token):
    
        headers = {"Authorization": 'Bearer ' + token,'tenantid': self.tenantid, 'tenantkey': self.tenantkey}
        data = {"name": name, "lastname": lastname, "details": details, "idIn": idIn}

        respAdd = requests.post(self.urlLocal + 'kycfaceid/v1/user/create',headers = headers,data = data)
        print(respAdd)
        return json.loads(respAdd.text)

    def addFace(self, frame, userId, token):
        
        cv2.imwrite('addface.png',frame)
        baseUri =""

        with open("addface.png", "rb") as imageFile:
            encoded = base64.b64encode(imageFile.read()).decode()
            baseUri = 'data:image/png;base64,{}'.format(encoded)

        headers = {"Authorization": 'Bearer ' + token,'tenantid': self.tenantid, 'tenantkey': self.tenantkey}
        dataReco = {"image": baseUri, "userId": userId}
    
        respReco = requests.post(self.urlLocal + 'kycfaceid/v1/user/addface',headers = headers,data = dataReco)


        return json.loads(respReco.text)
