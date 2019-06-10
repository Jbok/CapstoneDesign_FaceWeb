from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./serviceAccountKey.json')
default_app = firebase_admin.initialize_app(cred,{
     'databaseURL': 'https://lucky-aed54.firebaseio.com/'
})




def add_data(uid, currentData):
     
     date = str(datetime.today().year) +'p'+ str(datetime.today().month) +'p'+ str(datetime.today().day)
     path = "/info/" + uid + "/" + date +"/"    
     
     ref = db.reference(path)
     ref.set(currentData)

     avgRef = db.reference("/info/" + uid + "/avg/")
     avgValue = (float(currentData['jaw']) + float(currentData['lips']) + float(currentData['nose'])) / 3
     
     userEmail = db.reference("/users/" + uid + "/email/").get()

     rankRefAvg = db.reference("/RANK/" + uid + "/Average/")
     rankRefEmail = db.reference("/RANK/" + uid + "/Email/")
     

     if avgRef.get() is None:
          avgRef.set(avgValue)
          rankRefEmail.set(userEmail)
          rankRefAvg.set(avgValue)
     else:
          if avgValue < avgRef.get():
               avgRef.set(avgValue)
               rankRefEmail.set(userEmail)
               rankRefAvg.set(avgValue)
          else:
               rankRefEmail.set(userEmail)
               rankRefAvg.set(avgRef.get())
               


def get_data(uid):

     ref2 = db.reference("/info/")

     date = str(datetime.today().year) +'p'+ str(datetime.today().month) +'p'+ str(datetime.today().day)
     path = "/info/" + uid + "/" + date + '/'

     ref3 = db.reference(path)
     if ref3 is not None:
          ref3.delete()
    
     history_result = ref2.get()
     return history_result
