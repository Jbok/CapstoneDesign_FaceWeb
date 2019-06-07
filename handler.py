import json
import facial_landmark
import firebase_function
import base64

# python AWS SDK
import boto3

BUCKET_NAME = 'lucky-faceweb-2019'
BUCKET_TRAINED_DATA_PATH = 'TrainedData/shape_predictor_68_face_landmarks.dat'
LOCAL_IMG_PATH = '/tmp/image.png'
LOCAL_TRAINED_DATA_PATH = '/tmp/train.dat'
'''
LOCAL_IMG_PATH = 'image.png'
LOCAL_TRAINED_DATA_PATH = 'train.dat'
'''

s3_client = boto3.client('s3')
s3_client.download_file(BUCKET_NAME, BUCKET_TRAINED_DATA_PATH, LOCAL_TRAINED_DATA_PATH)

def test(event, context):
    
    f = open('/tmp/log3.txt', "w")
    f.write(json.dumps(event))
    f.close()
    s3_client.upload_file('/tmp/log3.txt', BUCKET_NAME, 'log3.txt')

    f2 = open('/tmp/log4.txt', "w")
    f2.write(json.dumps(event))
    f2.close()
    s3_client.upload_file('/tmp/log4.txt', BUCKET_NAME, './log4.txt')

 
    img_data = json.dumps(event['body']['data'])

    fh = open(LOCAL_IMG_PATH, "wb")
    fh.write(base64.decodestring(img_data.encode()))
    fh.close()

    result_currentData = facial_landmark.cal_asymmetry(LOCAL_TRAINED_DATA_PATH, LOCAL_IMG_PATH)


    uid = json.dumps(event['body']['uid'])
    f4 = open('/tmp/uid.txt', "w")
    f4.write(json.dumps(event))
    f4.close()
    s3_client.upload_file('/tmp/uid.txt', BUCKET_NAME, './uid.txt')

    print(uid)
    print("2")
    result_historyData = firebase_function.get_data(uid)

    resultArr = []
    for key, value in result_historyData.items():
        resultArr.insert(0, json.dumps(key))

    print("Aaaa")
    print(resultArr)
    print("bb")
    print(json.dumps(result_historyData))
    print("cc")
    print(str(json.dumps(result_historyData)))
    print("dd")
    
    tempStr = str(json.dumps(result_historyData))
    print("cc")
    print(tempStr.strip('{}'))
    print("dd")
    print('[' + tempStr.strip('{}') + '}' + ']')

    firebase_function.add_data(uid, result_currentData)


    response = {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            },
        "data" : {"currentData": result_currentData, "histories": result_historyData},
	    "message" : "Hello"
    }
    print(response)

    return response
