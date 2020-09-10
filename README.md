# Deploying models on AWS Sagemaker

## Define a build image in Sagemaker
- Build model train Docker container using template app (according to [AWS Sagemaker specs](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html)). This repo contains a setup for a Docker container that is compliant with the specs.  

- Create a new repository in [AWS ECR](https://eu-west-1.console.aws.amazon.com/ecr/get-started?region=eu-west-1).   (Take care in selecting the correct region, this should be the same as your S3 resource bucket).

- Execute the push commands from 'View Push Commands'.

## Create a Sagemaker Algorithm Resource
- Upload your dataset to S3
- In Sagemaker, in the left side bar, find the "Algorithms" button.
- On the right, click the 'Create Algorithm' button.
- Find your instance URI (in ECR) and copy paste it into Training Image box.
- Select the appropriate instance type.
- Make sure to change the name of the 'train' channel to 'training' as this corresponds to the path we chose to find out data (`/opt/ml/input/training...`)./
- Skip specifying Hyperparameters screen (Next)
- For container inference, also use the URI of the image (the inference image is used to make predictions - we will use the same image for this purpose).
- In the next screen, select the instance type twice
- Press Create Instance.

## Training Jobs
- Click the algorithm resource, click "Create Training Job"
- Choose a name
- Execution role, default
- Choose to use "Your own algorithm resource"
- For some mysterious reason, we have to choose the instance again.
- Make sure to specify the timeout condition.
- Verify the training channel name.
- Configure the training channel, Add the S3 location (which is the **directory of** the training data, and not the file itself.
- Skip to Output data configuration, and add the desired S3 resource to output the model to.

# Inference through an API Endpoint
- The [AWS docs](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html) describe the requirements for an image that serves predictions through an endpoint.
- Your application must respond to a command `serve` that will then serve an endpoint `POST /invocations` that processes prediction requests.
- It must also listen to an endpoint `GET /ping`, with minimum requirement being to return status code 200. AWS uses this for health checking your endpoint.
- In the left side bar:
  - Create a model for your artifact (using the inference image and S3 resource containing the artifact).
  - Create an Endpoint (+Endpoint configuration) that serves model predictions.
- If you update your model, you can update your endpoint by changing the endpoint configuration. No need to recreate the endpoint itself.

Use `boto3` to query the endpoint from a Python script. AWS endpoints do not respond when you use the bare URL in a request (due to security).

```python
import boto3
import json
ENDPOINT_NAME = 'who_lives_or_dies'

passengers = [
        {'PassengerId': 1, 'Survived': 0, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0,
         'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': 'unknown', 'Embarked': 'S'},
        {'PassengerId': 2, 'Survived': 1, 'Pclass': 1, 'Name': 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
         'Sex': 'female', 'Age': 38.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'PC 17599', 'Fare': 71.2833, 'Cabin': 'C85',
         'Embarked': 'C'},
        {'PassengerId': 3, 'Survived': 1, 'Pclass': 3, 'Name': 'Heikkinen, Miss. Laina', 'Sex': 'female', 'Age': 26.0,
         'SibSp': 0, 'Parch': 0, 'Ticket': 'STON/O2. 3101282', 'Fare': 7.925, 'Cabin': 'unknown', 'Embarked': 'S'}
        ]
# boto fetches your aws configs from the command line 
# (you can configure using: aws configure ...         

runtime = boto3.client('runtime.sagemaker', region_name='eu-west-1')
response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='application/json',
    # The request body has a 'data' key. Of course this is 
    # entirely optional.
    Body=json.dumps({'data': passengers}).encode()
)

result = json.loads(response['Body'].read().decode())
print(result)
```
# Gotcha's:
- When `model.pkl` changes. You need to start from the beginning, redefing the Algorithm as well.
- When the data changes, make a new Training Job to create a new model.
- When the Model changes, make a new Endpoint Configuration.
- Apply the new Endpoint Configuration to the Endpoint.