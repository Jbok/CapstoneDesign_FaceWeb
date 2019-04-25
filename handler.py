import json
import facial_landmark

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
    s3_client.download_file(bucket_name, trained_data_path, local_file_path)


def test(event, context):
    
    img_data = json.dumps(event['body'])

    BUCKET_NAME = 'lucky-faceweb-2019'
    BUCKET_TRAINED_DATA_PATH = 'TrainedData/shape_predictor_68_face_landmarks.dat'
    LOCAL_IMG_PATH = '/tmp/image.png'
    LOCAL_TRAINED_DATA_PATH = '/tmp/train.dat'

    fh = open(LOCAL_IMG_PATH, "wb")
    fh.write(img_data.decode('base64'))
    fh.close()
    
    result = facial_landmark.cal_asymmetry(LOCAL_TRAINED_DATA_PATH, LOCAL_IMG_PATH)

    print(result)
