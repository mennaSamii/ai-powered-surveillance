import mysql.connector 
import datetime
import queue
import random
from datetime import date,datetime,time
incident_queue = queue.Queue()
db_config = {
'host': "127.0.0.1",
'user': 'root',
'password': '_Admine1234',
'database': 'SecuritySystem'
}
class IncidentManagement():
     
    def __init__(self,db_config):

        try:
            self.connection = mysql.connector.connect(**db_config)
            
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            self.connection = None  # Set to None to indicate connection failure
        #self.processframes()
        #self.db_connector = db_connector
         
    def ProcessIncident(self):
        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None 
        
        while True:
            incident=incident_queue.get()
            # Check for latest incident and its EndTime
            #print(type(incident)) 
            #if self.has_indices(incident):
             #   print (incident.get('incidents'))
            #print ("one incident only ")
            print ("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",incident)
            for i in incident.get('incidents'):
                action=[]
                action=i.get('type')
                print (action)
                content = action
                Camera_ID = i.get('Camera_ID')
                # Extract person names from the list
                person_names = []
                for face in i.get('json_model_FaceRecognition').get('Faces'):
                    person_names.append(face['person_name'])
                # Combine incident type and person names (adjust formatting as needed)
                    if person_names:
                        content += f" - Detected persons: {', '.join(person_names)}"
                    else:
                        content += " - No faces detected"

                    print(content)
                    cursor = self.connection.cursor()
                    sql = "SELECT ID FROM Incident where EndTime is null AND (IncidentType ='{}')".format(action)
                    cursor.execute(sql)
                    result = cursor.fetchone()
                    print("open incdent id",result)
                    #self.Location()
                    #SecurityPhoneNumber= '12345678910'
                    #self.CloseAlert(SecurityPhoneNumber,result)
                    #print ("EndTime",result)
                    #self.CreateAlert(content,result,Camera_ID)
                    if result is not None:
                        print ("ahooooooooo fy incident mafto7a\n")
                        #sql1 = "SELECT ID FROM Incident where EndTime is null"
                        #cursor.execute(sql1)
                        #result1 = cursor.fetchone()
                        frame_time =i.get('frame_time')
                        frame_path =i.get('frame_path')
                        InferenceJSON =i.get('json_model')
                        print (frame_path,frame_time)
                        frame_data = {
                            'CameraID':Camera_ID,
                            'frame_time': frame_time,
                            'frame_path':frame_path
                        }
                        self.UpdateFrames(frame_data)
                        print (self.frame_id)
                        incident_frame_data={
                            'FrameID':self.frame_id,
                            'IncidentID':result,
                            'InferenceJSON':InferenceJSON
                        }
                        
                        self.UpdateIncidentFrames(incident_frame_data)
                        #HENNA KAMN YA MENNA EZBOTYHA EFTKRYYY YA ANSA  
                        if i.get('json_model_FaceRecognition'):
                            person_involved = []
                            for face in i.get('json_model_FaceRecognition').get('Faces'):
                                person_name = face['person_name']
                                print(person_name)
                                person_involved.append(person_name)
                            self.IdentifyInvolvedPerson(person_involved,result)

                        else :
                            print ("cant detect the person")
                            return  result
                    elif result is None or result[0] is None:
                        # Prepare data for the new incident
                        print ("typeeeeeee",action )
                        incident_data = {
                            'CameraID':Camera_ID,
                            'StartTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'IncidentType':action
                        }
                        incident_id=self.CreateIncident(incident_data)
                        if incident_id:
                            print(f"New incident created with ID: {incident_id}")
                            print (incident_id)
                        self.CreateAlert(content,incident_id,Camera_ID)
                        
                    #incident_data = {
                    #'StartTime': '2024-07-27 14:55:00'}
                    #if frame_processor() is not None :
                
            incident_queue.task_done()
    

 
    def Location(self):
        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None
        DateOfToday= date.today().strftime('%Y-%m-%d')
        try:
            cursor = self.connection.cursor(buffered=True)
            # Check if a record already exists for the same date
            sql_check = "SELECT * FROM SecurityPersonnelLocation WHERE DateOfToday = %s"
            cursor.execute(sql_check, (DateOfToday,))
            existing_record = cursor.fetchone()
            if not existing_record:
                cursor = self.connection.cursor()
                sql ="SELECT ID FROM SecurityPersonnel"
                cursor.execute(sql)
                SecurityPersonnel_IDs = [row[0] for row in cursor.fetchall()]  # Assuming ID is the first column
                print("ids returned ",SecurityPersonnel_IDs)
                # Shuffle the list of IDs randomly
                random.shuffle(SecurityPersonnel_IDs)
                print("ids returned ",SecurityPersonnel_IDs)
                sql1 ="SELECT ID FROM Location"
                cursor.execute(sql1)
                Location_IDS=[row[0] for row in cursor.fetchall()]
                print("ids returned ",Location_IDS)
                random.shuffle(Location_IDS)
                print("ids returned ",Location_IDS)
                StartTime='09:00:00'
                EndTime='17:00:00'
                sql_insert="Insert INTO SecurityPersonnelLocation (LocationID,SecurityPersonnelID,StartTime,EndTime,DateOfToday) values (%s,%s,%s,%s,%s)"
                # Loop through the shuffled IDs and insert them into the table
                for security_id, location_id in zip(SecurityPersonnel_IDs,Location_IDS):
                    cursor.execute(sql_insert, ( location_id,security_id,StartTime,EndTime,DateOfToday))
                    self.connection.commit()
                print("Successfully inserted random ID pairs into the SecurityPersonnelLocation")
            else:
                print("Record already exists for", DateOfToday)
            
        except mysql.connector.Error as err:
            print(f"Error creating Incident: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None

    def GetLocation(self,SecurityPhoneNumber):
        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None

        cursor = self.connection.cursor()
        #date_of_today= date.today().strftime('%Y-%m-%d')

        try:
            #sql=" SELECT ID FROM SecurityPersonnel WHERE PhoneNumber= %s "
            #cursor.execute(sql,(SecurityPhoneNumber,))
            #SecurityPersonnel_ID = cursor.fetchone()
            #sql1="SELECT LocationID FROM SecurityPersonnelLocation WHERE (SecurityPersonnelID , DateOfToday) = (%s,%s) "
            #cursor.execute(sql1,SecurityPersonnel_ID,date.today())
            #LocationID = cursor.fetchone()
            #if LocationID:
                #sql2 = "SELECT Location FROM Location WHERE ID = %s"
                #cursor.execute(sql2, LocationID)
                #Location = cursor.fetchone()
                #return Location
            sql = """SELECT l.Location FROM SecurityPersonnel sp
                    INNER JOIN SecurityPersonnelLocation spl ON sp.ID = spl.SecurityPersonnelID
                    INNER JOIN Location l ON spl.LocationID = l.ID
                    WHERE sp.PhoneNumber = %s AND spl.DateOfToday = %s
                    """
            cursor.execute(sql, (SecurityPhoneNumber, date.today()))
            location = cursor.fetchone()#[0]
            if location:
                print("The location for this PhoneNumber: "+SecurityPhoneNumber+" is "+location[0])
                return location[0]
            else:
                print("No Location found for SecurityPersonnel with PhoneNumber:", SecurityPhoneNumber)
                return None
        except mysql.connector.Error as err:
            print(f"Error finding location : {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None

    def CreateIncident(self,data):

        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None

        cursor = self.connection.cursor()
        try:
            sql = "INSERT INTO Incident (CameraID,StartTime,IncidentType)VALUES (%s,%s,%s)"

            
            cursor.execute(sql, (data['CameraID'],data['StartTime'],data['IncidentType']))
            self.connection.commit()

            # Get the newly created incident ID
            self.incident_id = cursor.lastrowid
            self.connection.commit()
            return self.incident_id

        except mysql.connector.Error as err:
            print(f"Error creating Incident: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
    
    def CloseIncident(self,incident_id):

        if self.connection is None:
            print("Database connection not established. Cannot update incident.")
            return False

        cursor = self.connection.cursor()
        end_time =datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
        try:
            sql = "UPDATE Incident SET EndTime = %s WHERE ID = %s"
            cursor.execute(sql, (end_time, incident_id))
            self.connection.commit()
            return cursor.rowcount > 0  # Check if at least one row was affected

        except mysql.connector.Error as err:
            print(f"Error updating incident: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return False
 
    def IdentifyInvolvedPerson(self,person_involved,result):
        if self.connection is None:
            print("Database connection not established. Cannot update incident.")
            return False
        #person_involved=[]
        print(person_involved)
        person_ids = []
        cursor = self.connection.cursor()
        
        try:
            for name in person_involved:
                sql = "SELECT ID FROM Person WHERE FirstName = '{}'".format(name)
                cursor.execute(sql)
                person_id = cursor.fetchone()
                if person_id:
                    person_ids.append(person_id)
                else:
                    print(f"Person not found: {name}")
            #sql ="SELECT ID FROM Person WHERE FirstName = '{}'".format(person_involved)
            #cursor.execute(sql)
            #self.person_id=cursor.fetchone()
            #print("person id",self.person_id)
            if person_ids :
                print ('person id',person_ids)
                IncidentInvolvedPerson={
                    'IncidentID':result,
                    'PersonID':person_ids
                }
                
                self.UpdateIncidentInvolvedPerson(IncidentInvolvedPerson)
                return person_ids
            else :
                return None
        except mysql.connector.Error as err:
            print(f"Error finding person id : {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
    
    def UpdateIncidentInvolvedPerson(self,data) :
        #print(data['PersonID'])
        #print(type(data['PersonID']))
        if self.connection is None:
            print("Database connection not established. Cannot update incident.")
            return False
        cursor = self.connection.cursor()
        try:
            for id in data['PersonID']:
                person_id=id[0]
                sql ="INSERT INTO IncidentInvolvedPerson (IncidentID,PersonID) VALUES (%s,%s)"
                cursor.execute(sql, (data.get('IncidentID')[0],person_id))
                self.connection.commit()
                print ("IncidentInvolvedPerson is updated successfully :)")
            
        except mysql.connector.Error as err:
            print(f"Error finding person id : {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
           

    def UpdateFrames(self,data):
        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None

        cursor = self.connection.cursor()
                # Retrieve the recently created incident ID (assuming self.incident_id is set)
        try:
            print("frame time ",data['frame_time'])
            # Prepare SQL statement with placeholders for data
            sql = "INSERT INTO Frame (CameraID, FrameTime, FramePath) VALUES (%s, %s, %s)"

            # Insert frame data along with the incident ID
            cursor.execute(sql, (data['CameraID'], data['frame_time'], data['frame_path']))
            self.connection.commit()  # Commit the changes to the database
            print("Frame data inserted successfully!")
            self.connection.commit()

            # Get the newly created incident ID
            self.frame_id = cursor.lastrowid


            self.connection.commit()
            return self.frame_id
            
        except mysql.connector.Error as err:
            print(f"Error creating Incident: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
    
    def UpdateIncidentFrames(self,data):
        #print (data['InferenceJSON'])
        #print (type("".join(data['InferenceJSON'])))
        if self.connection is None:
            print("Database connection not established. Cannot create incident.")
            return None

        cursor = self.connection.cursor()
                # Retrieve the recently created incident ID (assuming self.incident_id is set)
        try:
            # Prepare SQL statement with placeholders for data
            sql = "INSERT INTO IncidentFrame (FrameID, IncidentID, InferenceJSON) VALUES (%s, %s, %s)"
            # Insert frame data along with the incident ID
            cursor.execute(sql, (data.get('FrameID'), data.get('IncidentID')[0],"".join(data.get('InferenceJSON')) ))
            # Commit the changes to the database
            self.connection.commit()  
            print("FrameIncident data inserted successfully!")
            self.connection.commit()
            return self.frame_id
    
            
        except mysql.connector.Error as err:
            print(f"Error creating Incident: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
        
    def CreateAlert (self,content,result,Camera_ID):
        """
        Creates a new alert record associated with an existing incident.

        Args:
            incident_id (int): The ID of the incident to associate with the alert.
            content (str): The content of the alert message.

        Returns:
            int: The ID of the newly created alert record, or None on error.
        """
        print (content)
        #print (type(Camera_ID))
        if self.connection is None:
            print("Database connection not established. Cannot create alert.")
            return None

        cursor = self.connection.cursor()
        #cursor = self.connection.cursor(buffered=True)
        try:
            sql = """SELECT l.Location FROM Camera c 
                    INNER JOIN Location l ON l.ID = c.LocationId
                    WHERE c.ID =%s"""
            #location=cursor.execute(sql, (Camera_ID )  
            print ("camera id",Camera_ID,"ahooo")      
            cursor.execute(sql,(Camera_ID,))
            m=cursor.fetchone()
            print (m)
            if m is not None:
                location = m[0]
                content = "An incident occurred at " + location + " now - " + content
                print(content )
            else:
                print("No location found for camera ID:",Camera_ID)
                # Consider setting a default location or handling the 
                
            sql1="INSERT INTO Alert (content,IncidentID) VALUES (%s,%s)"
            alert=cursor.execute(sql1, (content,result)) 
            self.connection.commit()
            print ('alert created',alert)
            self.SendAlert(result,content)

        except mysql.connector.Error as err:
            print(f"Error creating alert: {err}")
        finally:
            cursor.close()  # Always close the cursor
        return None

    def SendAlert(self,result,content):
        
        if self.connection is None:
            print("Database connection not established. Cannot create alert.")
            return None

        cursor = self.connection.cursor()
        DateOfToday=date.today().strftime('%Y-%m-%d')
        try:
            sql = """
            SELECT spl.SecurityPersonnelID, spl.StartTime, spl.EndTime, sp.PhoneNumber
            FROM Incident i
            INNER JOIN Camera c ON i.CameraID = c.ID
            INNER JOIN SecurityPersonnelLocation spl ON c.Locationid = spl.LocationID
            INNER JOIN SecurityPersonnel sp ON spl.SecurityPersonnelID = sp.ID
            WHERE i.ID = %s AND spl.DateOfToday = %s;
            """
            cursor.execute(sql, (result, DateOfToday))
            security_personnel_data = cursor.fetchone()
            print("hiii",security_personnel_data)
            start_time=security_personnel_data[1].total_seconds() % (24 * 3600)
            end_time=security_personnel_data[2].total_seconds() % (24 * 3600)
            current_time = datetime.now().time().hour * 3600 + datetime.now().time().minute * 60
            if start_time <= current_time <=end_time:
                sql4="select PhoneNumber from SecurityPersonnel where ID =%s "
                cursor.execute(sql4,(security_personnel_data[0],))
                
                PhoneNumber = cursor.fetchone()
                print("after linking with chatboot the message will be send to this number ",PhoneNumber[0])
                return PhoneNumber
            print(result,content)
            return result,content
        except mysql.connector.Error as err:
            print(f"Error sending  alert: {err}")
        finally:
            cursor.close()  # Always close the cursor

        return None
    


    
    def CloseAlert(self,SecurityPhoneNumber,result):
        if self.connection is None:
            print("Database connection not established. Cannot create alert.")
            return None
        cursor = self.connection.cursor()
        StatusOfAlert='closed'
        ConfirmationDate= datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            #el mafrod hna kman ha nupdate el sequrity personelid 
            sql = """UPDATE Alert SET SecurityPersonnelID = (SELECT ID FROM SecurityPersonnel WHERE PhoneNumber = %s),
                     StatusOfAlert = %s, ConfirmationDate = %s  
                     WHERE (IncidentID)  = %s 
                    """
            cursor.execute(sql, (SecurityPhoneNumber,StatusOfAlert,ConfirmationDate,result[0]))
            self.connection.commit()
            #return cursor.lastrowid

        except mysql.connector.Error as err:
            print(f"Error updating alert: {err}")
        finally:
            cursor.close()  # Always close the cursor
        return None
        
    #def addCommentToIncident(self):
    #def closeAlert(self):
if __name__ == '__main__':
    Camera_ID=0
    # Prepare data for the new incident
    incident_data = {
    'CameraID':Camera_ID,
    'StartTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'IncidentType':'fall down'
    }
    SecurityPhoneNumber='01045310463'
    inc= IncidentManagement(db_config)
    action ='fall down - Detected persons: menna'
    result='6'
    #inc.CreateIncident(incident_data)
    inc.CloseIncident(15)
    inc.Location()
    #inc.CreateAlert(action,result,Camera_ID)
    #inc.SendAlert(result)
    #inc.GetLocation(SecurityPhoneNumber)
    #incident_id=inc.CreateIncident(incident_data)
    #inc.CloseAlert(SecurityPhoneNumber,result)
    #if incident_id:
        #   print(f"New incident created with ID: {incident_id}")

    # Assuming you want to close the incident with a specific end time
    # end_time = '2024-07-27 15:00:00'  # Replace with the actual end time
    #closed = inc.CloseIncident(end_time)

    #if closed:
        #   print(f"Incident ID {incident_id} successfully closed.")
    #else:
    #    print(f"Failed to close incident ID {incident_id}.")"""
    #inc.closeIncident(incedent_close)"""