import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./local_test_data/serviceAccountKey.json')
default_app = firebase_admin.initialize_app(cred,{
     'databaseURL': 'https://faceweb-2d9dd.firebaseio.com/'
})

tempUserID = "/tu0/"
ref = db.reference(tempUserID)
ref.set({
  "tableData": [
    {
      "UserID": "Damon",
      "순위": 1,
      "평균": "11.1111"
    },
    {
      "UserID": "Damon",
      "순위": 3,
      "평균": "22.2222"
    },
    {
      "UserID": "Damon",
      "순위": 2,
      "평균": "33.3333"
    }
  ]
})