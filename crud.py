import psycopg2
from psycopg2.extras import RealDictCursor

class CRUD:


  def __init__(self):
    self.strinConn ="user=kycfaceuser password=face2018 host=localhost port=5432 dbname=kycfaceDB"
 
  def returnData(self,queryString):
    connection=None
    cursor = None
    
    try:        
      connection = psycopg2.connect(self.strinConn)
      cursor = connection.cursor(cursor_factory=RealDictCursor)
      cursor.execute(queryString)
      records = cursor.fetchall()
      

      return records

    except (Exception, psycopg2.Error) as error:
      return error
  
    finally:     
      if(connection):
        cursor.close()
        connection.close()
           
  def insertData(self,insertString,records):
    connection=None
    cursor = None
    
    try:
     
      connection = psycopg2.connect(self.strinConn)
      cursor = connection.cursor()  
      if(records==None):
        cursor.execute(insertString)
      else:
        cursor.execute(insertString, records)              
      
      cursor.execute('SELECT LASTVAL()')
      lastid = cursor.fetchone()[0]
      connection.commit()
     		   		   
   	
      return lastid
   
    except (Exception, psycopg2.Error) as error:
      return "Error"

    finally:
      if(connection):
        cursor.close()
        connection.close()

  def updateData(updateString,records):
    connection=None
    cursor = None

    try:
      connection = psycopg2.connect(strinConn)
      cursor = connection.cursor()
      cursor.execute(updateString, records)
      connection.commit()
      count = cursor.rowcount

      return count
   		
    except (Exception, psycopg2.Error) as error:
      return error.diag.message_detail

    finally:
      if(connection):
        cursor.close()
        connection.close()

  def deleteData(deleteString,records):
    connection=None
    cursor = None

    try:
      connection = psycopg2.connect(strinConn)
      cursor = connection.cursor()
      cursor.execute(deleteString, records)
      connection.commit()
      count = cursor.rowcount

      return count
   		
    except (Exception, psycopg2.Error) as error:
    	return error.diag.message_detail

    finally:
      if(connection):
        cursor.close()
        connection.close()
