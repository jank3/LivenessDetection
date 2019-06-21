import sys
import dlib
import cv2
import face_recognition
import os
import psycopg2
import crud
import io,base64
import re
from datetime import date

crudOperation = crud.CRUD()

class encod:

	def encoding(self,file_name,faceId,tenantid):
		data = None
		query=""
		entry_encoding=""

		try:
			retorna="e1"
			#face_detector = dlib.get_frontal_face_detector()			
			retorna="e2"
			#image = cv2.imread(file_name)
			retorna="e3"
			#detected_faces = face_detector(image, 1)
			retorna="e4"
			image_entry = face_recognition.load_image_file(file_name)
			retorna="e5"
			entry_encoding = face_recognition.face_encodings(image_entry)[0]
			retorna="e6"
			
			if len(entry_encoding) > 0:
				retorna="e7"
				query = """insert into kycface."vectors"("faceId","tenantid","vec_low", "vec_high") VALUES('{}','{}', CUBE(array[{}]), CUBE(array[{}]))""".format(
					faceId,tenantid,','.join(str(s) for s in entry_encoding[0:64]),
					','.join(str(s) for s in entry_encoding[64:128]),)
				
				retorna="e8"
				data = crudOperation.insertData(query,None)

			#cv2.imwrite("/var/www/flask/face/faces/aligned_face_{}_{}_crop.jpg".format(faceId, i), crop)


			#for i, face_rect in enumerate(detected_faces):
				# crop = image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
				# encodings = face_recognition.face_encodings(crop)

				# if len(encodings) > 0:
				# 	query = """insert into kycface."vectors" ("faceId", "vec_low", "vec_high") VALUES ('{}', CUBE(array[{}]), CUBE(array[{}]))""".format(
				# 		faceId,
				# 		','.join(str(s) for s in encodings[0][0:64]),
				# 		','.join(str(s) for s in encodings[0][64:128]),
				# 		)

				# 	data = crudOperation.insertData(query,None)

				# cv2.imwrite("/var/www/flask/face/faces/aligned_face_{}_{}_crop.jpg".format(faceId, i), crop)

		except Exception as e:
 			return str(e)
		
		return data
