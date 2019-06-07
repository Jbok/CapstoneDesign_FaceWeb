from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./serviceAccountKey.json')
default_app = firebase_admin.initialize_app(cred,{
     'databaseURL': 'https://faceweb-2d9dd.firebaseio.com/'
})


def add_data(uid, currentData):
    date = str(datetime.today().year) +'p'+ str(datetime.today().month) +'p'+ str(datetime.today().day)
    path = "/" + uid + "/" + date +"/"    
    ref = db.reference(path)
    ref.set(currentData)

def get_data(uid):
    ref2 = db.reference('/' + uid)

    date = str(datetime.today().year) +'p'+ str(datetime.today().month) +'p'+ str(datetime.today().day)
    path = "/" + uid + "/" + date +"/"
    ref3 = db.reference(path)
    ref3.remove()
    

    history_result = ref2.get()
    print(history_result)
    return history_result
