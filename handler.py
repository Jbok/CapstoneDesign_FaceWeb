import json
import numpy as np
import facial_landmark

def hello(event, context):

    print(np.arrange(15).reshape(3, 5))

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
    result = facial_landmark.cal_assymetry('shape_predictor_68_face_landmarks.dat','balance.png')
    print(result)