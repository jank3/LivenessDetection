import requests
import os
import io
import json
import jsonify
#Libreria de face_recognition, para realizar el test
import face_recognition
import crud
import datetime
import encode_face
import recognize_faces_image
import psycopg2 as psql
import numpy as np
import traceback
from flask_jwt_extended import create_access_token

crudOperation = crud.CRUD()
faceEncode = encode_face.encod()
faceRecognizaImage = recognize_faces_image.matchFace()

class kycfaceid:

  #Funciones Group 
  def groupAddUser(groupId, userId,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupId must be a string"}
    
    try:
      dt = datetime.datetime.now();
      lastid = crudOperation.insertData("""insert into kycface."userGroup"("groupId","userId","dateCreate","tenantid") values(%s,%s,%s,%s)""",(groupId,userId,dt,tenantid))
      
      retorno = None

      if(lastid<=0):
        retorno = False
      else:
        retorno = True

      data = {
        'result': retorno        
       }

    except requests.exceptions.RequestException as e:
      return e


    return data;

  def groupCreate(groupName,sizeLimit,tenantid):
    if(type(groupName) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}
    if(type(sizeLimit) is not str): return {"error" : "invalid_input", "message" : "sizeLimit must be a string"}
      
    try:
      dt = datetime.datetime.now();

      lastid = crudOperation.insertData("""insert into kycface."groupTb"("groupName","sizeLimit","state","dateCreate","tenantid") values(%s,%s,%s,%s,%s)""",(groupName,sizeLimit,0,dt,tenantid))
     
    except requests.exceptions.RequestException as e:
      return e
    
    return lastid;

  def groupEdit(groupId,groupName,sizeLimit,tenantid):
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupId must be a string"}
    if(type(groupName) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}
    if(type(sizeLimit) is not str): return {"error" : "invalid_input", "message" : "sizeLimit must be a string"}
            
    try:
      rets=None

      rowsUpdate = crudOperation.updateData("""update kycface."groupTb" set "groupName"=%s,"sizeLimit"=%s where "groupId"=%s and "tenantid"=%s """,(groupName,sizeLimit,groupId,tenantid))

      if(lastid<=0):
        rets = False
      else:
        rets = True

      data = {
        'result': rets        
       } 

    except requests.exceptions.RequestException as e:
      return e

    return data;

  def groupGet(groupId,tenantid):
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}
      
    try:
        
        dataGroup = crudOperation.returnData("""select "groupName","sizeLimit" from kycface."groupTb" where "groupId"=""" + groupId + """ and "tenantid"=""" + tenantid)

        dataUser = crudOperation.returnData("""select "userId","groupRole" from kycface."userRole" where "groupId"=""" + groupId + """ and "tenantid"=""" + tenantid)
        
        data = {
          'result':{
            dataGroup,dataUser
          }
        }

    except requests.exceptions.RequestException as e:
      return e
    return data;

  def groupList(tenantid):

    try:
      dataGroups = crudOperation.returnData("""select "groupId","groupName","sizeLimit" from kycface."groupTb" where "state"=0 and "tenantid"=""" + tenantid)
      
      data = {
        'result':{
         'groups': dataGroups
        }
      }

    except requests.exceptions.RequestException as e:
      return e
    return data;

  def groupListUser(userId,tenantid):
    try:
      dataUsers = crudOperation.returnData("""select a."groupId",a."groupName",a."sizeLimit" from kycface."groupTb" as a, kycface."userGroup" as b where b."userId"=""" + userId + """ and a."groupId" = b."groupId" and  a."state"=0 and a."tenantid"=""" + tenantid)
      
      data = {
      'result': dataUsers
      }

    except requests.exceptions.RequestException as e:
      return e
    return data;

  def groupRemove(groupId,tenantid):
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}

    try:
      rets = None
      deleteGroupUser = crudOperation.returnData("""delete from kycface."userGroup" where "groupId"=%s and "tenantid"=%s""",(groupId,tenantid))
      deleteGroup = crudOperation.returnData("""delete from kycface."groupTb" where "groupId"=%s and "tenantid"=%s""",(groupId,tenantid))
      
      if(deleteGroup > 0) and (deleteGroupUser >= 0):
        rets = True
      else:
        rets = False
        
      data = {
        'result': rets        
       }
    except requests.exceptions.RequestException as e:
      return e
    return data;

  
  def groupRemoveUser(groupId,userId,tenantid):
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}
      
    try:
      rets = None
      deleteGroup = crudOperation.returnData("""delete from kycface."userGroup" where "userId"=%d and "groupId"=%s and "tenantid"=%s  """,(userId,groupId,tenantid))
      
      if(deleteGroup > 0):
        rets = True
      else:
        rets = False

      data = {
        'result': rets        
       }
    except requests.exceptions.RequestException as e:
      return e
    return data;

  def groupUserRole(groupId,userId,groupRole,tenantid):
    if(type(groupId) is not str): return {"error" : "invalid_input", "message" : "groupName must be a string"}
      
    try:
      dt = datetime.datetime.now();
      lastid = crudOperation.insertData("""insert into kycface."userRole"("groupId","userId","groupRole","tenantid") values(%s,%s,%s,%s)""",(groupId,userId,groupRole,tenantid))
      
      rets = None

      if(lastid<=0):
        rets = False
      else:
        rets = True

      data = {
        'result': rets        
       }
    
    except requests.exceptions.RequestException as e:
      return e
    return data;


#Funciones User
  def userAddFace(faceFilePath,userId,tenantid):
    if(type(userId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}
    rets=None 

    try:      
      dt = datetime.datetime.now();      
      lastid = crudOperation.insertData("""insert into kycface."userFace"("userId","image","dateCreate","tenantid") values(%s,%s,%s,%s)""",(userId,'data',dt,tenantid))      
      codigo = faceEncode.encoding(faceFilePath,lastid,tenantid)
      
      rets ={'return': {'faceId': lastid}}

    except Exception as e:
      return str(e)

    return lastid

  def userAuth(userId,otp,tenantid):
    if((not os.path.exists(faceFilePath))): return {"error":"archivo no valido", "mensaje" : faceFilePath+" no existe"}
    if(type(userId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}
    if(type(permanent) is not str): return {"error" : "entra no valida", "mensaje" : "permanent tiene que ser texto"}

    access_token=None

    try:
      
      dataUser = crudOperation.returnData("""select "userId", "token" from kycface."userTb" where "name"='""" + name + """' and "token"='""" + otp + """' and "tenantid"=""" + tenantid )

      if (dataUser > 0 ):
        access_token = create_access_token(identity=otp)
      else:
        access_token=None


      data = {'result': {'token': access_token}}
    

    except Exception  as e:
      return str(e)

    return data

  def userCreate(name,lastname,details,idIn,tenantid):
    if(type(name) is not str): return {"error" : "invalid_input", "message" : "Name must be a string"}
    if(type(lastname) is not str): return {"error" : "invalid_input", "message" : "LastName must be a string"}
    if(type(details) is not str): return {"error" : "invalid_input", "message" : "details must be a string"}
  
    rets=None
    retorna="1"
    try:
      retorna="2"
      dt = datetime.datetime.now();
      retorna="3"
      lastid = crudOperation.insertData("""insert into kycface."userTb"("name","lastname","details","dateCreate","state","token","tenantid") values(%s,%s,%s,%s,%s,%s,%s)""",(str(name),str(lastname),str(details),dt,0,str(idIn),tenantid))
      retorna="4"
      if (lastid>0):
        retorna="5"
        rets=lastid
      else:
        retorna="6"
        rets=None

      retorna="7"
      data = {'result': rets}

    except Exception as e:
      return str(retorna)

    return data

  def userEdit(userId,name,LastName,phone,details,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}
    if(type(details) is not str): return {"error" : "invalid_input", "message" : "details must be a string"}

    rets=None

    try: 
      rowsUpdate = crudOperation.updateData("""update kycface."userTb" set "name"=%s,"lastName"=%s,"phone"=%s,"details"=%s where "userId"=%s and "tenantid"=%s """,
      (name,lastName,phone,details,userId,tenantid))

      if(rowsUpdate > 0):
        rets = True
      else:
        rets = False 

      data = {
        'result': rets        
       }

    except Exception as e:
      return str(e)

    return data

  def userEnroll(userId, details,groupId,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}
    if(type(details) is not str): return {"error" : "invalid_input", "message" : "details must be a string"}

    try:
      data = {
        'faceIds': [{'label' : 'the-key-name-for-file','faceId' : 'string-max-length-40-chars'}]
        }      
       
    except Exception as e:
      return str(e)

    return response

  def userFaceAuth(faceFilePath,tenantid):
    retorna=""
    data = None
    if(type(userId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}

    try:

      faceId=""
      rets = faceRecognizaImage.searchMatch(faceFilePath)      

      if not rets:
        return {"message":"Face without Recognition"}

      for row in rets:
        faceId = row["faceId"]

      retsUser = crudOperation.returnData("""select "groupId", "groupName" from kycface."groupFromFaceVw" where "idFace"=""" + str(faceId) + """ and "tenantid"=""" + tenantid)
            
      data ={'result':{retsUser}}

    except Exception as e:
      return retorna

    return data

  def userGet(userId,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}    

    try:
      retsUser = crudOperation.returnData("""select "userId","details","name","lastName","phone" from kycface."userTb" where "userId"=""" + str(userId) + """ and "tenantid"=""" + tenantid)
      retsFaces = crudOperation.returnData("""select "faceid" from kycface."userFace" where "userId"=""" + str(userId) + """ and "tenantid"=""" + tenantid)      
      retsGroup = crudOperation.returnData("""select a."groupId",a."groupName",a."sizeLimit" from kycface."groupTb" as a, kycface."userGroup" as b
      where b."userId"=""" + userId + """ and a."groupId" = b."groupId" and  a."state"=0""" + """ and "tenantid"=""" + tenantid)
      
     
      data = {'result': {'userId': retsUser, 'faceId': retsFaces,
             "roles": ['user'],'groups': retsGroup
        }}      
       
    except Exception as e:
      return str(e)

    return data

  def userGetManual(name,lastname,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}    

    try:
         
      data = crudOperation.returnData("""select "userId", "name", "lastname","details","token" from kycface."userTb" where "name"='""" + name + """' and "token"='""" + phone + """' """ + """ and "tenantid"=""" + tenantid )

      return data
    
    except Exception as e:
      return str(e)


  def userGetOTP(userId,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}    

    try:
     data = {
        'result': true        
       }
    except Exception as e:
      return str(e)

    return response

  def userRemove(userId,tenantid):
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}   
    
    rets=None

    try: 
      rowsUpdate = crudOperation.updateData("""update kycface."userTb" set "state"=1 where "userId"=%s and "tenantid"=%s""",
      (userId,tenantid))

      if(rowsUpdate > 0):
        rets = True
      else:
        rets = False 

      data = {
        'result': rets        
       }
    except Exception as e:
      return str(e)

    return response

  def userRemoveFace(userId,faceId,tenantid):
    if(type(faceId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}
    if(type(userId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}

    rets = None
    try:

      lastid = crudOperation.insertData("""delete from kycface."userFace" where userId=%s and idFace=%s and "tenantid"=%s""",(userId,faceId,tenantid))

      if (lastid>0):
        rets = True
      else:
        rets = False

      data = {'result': rets}
    
    except Exception  as e:
      return str(e)

    return data

  def userRole(roles,userId,tenantid):
    if(type(roles) is not str): return {"error" : "invalid_input", "message" : "userId must be a string"}
    if(type(userId) is not str): return {"error" : "invalid_input", "message" : "details must be a string"}

    try:

      lastid = crudOperation.insertData("""insert into kycface."userRole"("userId","groupRole","tenantid") values(%s,%s,%s)""",(userId,groupRole,tenantid))
      
      if (lastid>0):
        rets=lastid
      else:
        rets=None

      data = {
        'result': true        
       }
    except Exception as e:
      return str(e)

    return response

  def userList(tenantid):
    retorno=dict()
    try:
      
      data = crudOperation.returnData("""select "userId", "name" || ' ' || "lastname" as nameFull,"details" from kycface."userTb" where "state"=0 and "tenantid"=""" + tenantid)
      

    except requests.exceptions.RequestException as e:
      return e
    return data;

#Funciones image
  def recognize(image_data,tenantid):   
    try:

      data = faceRecognizaImage.searchMatch(image_data)      
      faceId=""
      
      if not data:
        return {"message":"rostro sin Reconocimiento"}

      for row in data:
        faceId = row["faceId"]

      dataO = crudOperation.returnData("""select "name","token" from kycface."userFaceAuthVw" where "idFace"=""" + str(faceId) + """ and "tenantid"=""" + tenantid)
           
      return dataO
    
    except Exception as e:
      return e.args
   
  def verify(faceFilePath,userId,tenantid):
    if((not os.path.exists(faceFilePath))): return {"error":"archivo no valido", "mensaje" : faceFilePath+" no existe"}
    if(type(userId) is not str): return {"error" : "entra no valida", "mensaje" : "userId tiene que ser texto"}
    rets = None
    dataret =""

    try:
   
      data = faceRecognizaImage.searchMatch(image_data)      
      faceId=""
      
      if not data:
        return {"message":"rostro sin Reconocimiento"}

      for row in data:
        faceId = row["faceId"]

      dataO = crudOperation.returnData("""select "name","fono" from kycface."userFaceAuthVw" where "idFace"=""" + str(faceId) + """ and "userId"=""" + userId + """ and "tenantid"=""" + tenantid)
      
      if(dataO > 0):
        rets = True
      else:
        rets = False

      dataret = {'result': rets}
    
      return 
    

    except Exception  as e:
      return str(e)

    return dataret

  def veriryTenant(tenantid,tenantkey):
    try:

      data = crudOperation.returnData("""select * from kycface."client" where "tenantid"=""" + tenantid + """ and "tenantkey"='""" + tenantkey + """' """ )

      return data

    except Exception as e:
      return str(e)




