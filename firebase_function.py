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

     userNameRef = db.reference("/users/" + uid + "/username/").get()
     userName = ""

     if userNameRef is None:
          userName = db.reference("/users/" + uid + "/displayName/").get()
     else:
          userName = db.reference("/users/" + uid + "/username/").get()

     rankRefAvg = db.reference("/RANK/" + uid + "/Average/")
     rankRefEmail = db.reference("/RANK/" + uid + "/Email/")
     rankRefName = db.reference("/RANK/" + uid + "/Name/")

     if avgRef.get() is None:
          print("1111111111111")
          avgRef.set(avgValue)
          rankRefEmail.set(userEmail)
          rankRefAvg.set(avgValue)
          rankRefName.set(userName)
     else:
          if avgValue < avgRef.get():
               print("2222222222222222")
               avgRef.set(avgValue)
               rankRefEmail.set(userEmail)
               rankRefAvg.set(avgValue)
               rankRefName.set(userName)
          else:
               print("33333333333333333")
               rankRefEmail.set(userEmail)
               rankRefAvg.set(avgRef.get())
               print('userName')
               print(userName)
               rankRefName.set(userName)