import sys
import dlib
import cv2
import face_recognition
import os
import psycopg2
import crud

crudOperation = crud.CRUD()

class matchFace:
    def searchMatch(self,file_name):
        data = None
        query=""
        entry_encoding=""

        try:
            #face_detector = dlib.get_frontal_face_detector()
            #detected_faces = face_detector(io.BytesIO(file_name), 1)
            threshold = 0.6
            retorna="r1"
            image_entry = face_recognition.load_image_file(file_name)
            retorna="r2"
            entry_encoding = face_recognition.face_encodings(image_entry)[0]
            retorna="r3"
            #print(len(entry_encoding))
            if(len(entry_encoding) > 0):
                retorna="r4"
                query = """select "faceId","tenantid" from kycface."vectors" where sqrt(power(CUBE(array[{}]) <-> "vec_low", 2) + power(CUBE(array[{}]) <-> "vec_high", 2)) <= {} """.format(
                    ','.join(str(s) for s in entry_encoding[0:64]),
                    ','.join(str(s) for s in entry_encoding[64:128]),
                    threshold,
                    ) + \
                    """order by sqrt(power(CUBE(array[{}]) <-> "vec_low", 2) + power(CUBE(array[{}]) <-> "vec_high", 2)) ASC LIMIT 1""".format(
                    ','.join(str(s) for s in entry_encoding[0:64]),
                    ','.join(str(s) for s in entry_encoding[64:128]),
                    )
                #print(query)
                retorna="r5"
                data = crudOperation.returnData(query)

                
                
        except Exception as e:
            return str(retorna)
        
        return data
        

        # for i, face_rect in enumerate(detected_faces):
        #     crop = image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
        #     encodings = face_recognition.face_encodings(crop)
        #     threshold = 0.6

        #     if(len(encodings) > 0):
        #         query = """select "faceId" from kycface."vectors" where sqrt(power(CUBE(array[{}]) <-> "vec_low", 2) + power(CUBE(array[{}]) <-> "vec_high", 2)) <= {} """.format(
        #             ','.join(str(s) for s in encodings[0][0:64]),
        #             ','.join(str(s) for s in encodings[0][64:128]),
        #             threshold,
        #             ) + \
        #         """order by sqrt(power(CUBE(array[{}]) <-> "vec_low", 2) + power(CUBE(array[{}]) <-> "vec_high", 2)) ASC LIMIT 1""".format(
        #             ','.join(str(s) for s in encodings[0][0:64]),
        #             ','.join(str(s) for s in encodings[0][64:128]),
        #         )
              
        #         data = crudOperation.returnData(query)
        
        #         return data
        #     else:
        #         return None
