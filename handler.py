import json
import facial_landmark
import base64

# python AWS SDK
import boto3

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

def download_trained_data(bucket_name, trained_data_path, local_file_path):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, trained_data_path, local_file_path)


def test(event, context):
    
    img_data = json.dumps(event['body'])

    print(type(img_data))
    print(type(img_data.encode()))

    BUCKET_NAME = 'lucky-faceweb-2019'
    BUCKET_TRAINED_DATA_PATH = 'TrainedData/shape_predictor_68_face_landmarks.dat'
    # LOCAL_IMG_PATH = '/tmp/image.png'
    # LOCAL_TRAINED_DATA_PATH = '/tmp/train.dat'
    LOCAL_IMG_PATH = 'image.png'
    LOCAL_TRAINED_DATA_PATH = 'train.dat'

    fh = open(LOCAL_IMG_PATH, "wb")
    fh.write(base64.decodestring(img_data.encode()))
    fh.close()
    
    download_trained_data(BUCKET_NAME, BUCKET_TRAINED_DATA_PATH, LOCAL_TRAINED_DATA_PATH)

    result = facial_landmark.cal_asymmetry(LOCAL_TRAINED_DATA_PATH, LOCAL_IMG_PATH)

    print(result)





