import json
import facial_landmark
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

def hello(event, context):

    
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

def test(event, context):
    
    img_data = json.dumps(event['body'])

    fh = open(LOCAL_IMG_PATH, "wb")
    fh.write(base64.decodestring(img_data.encode()))
    fh.close()

    result = facial_landmark.cal_asymmetry(LOCAL_TRAINED_DATA_PATH, LOCAL_IMG_PATH)

    print(result)





