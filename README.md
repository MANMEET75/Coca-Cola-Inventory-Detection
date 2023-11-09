# Coca Cola Inventory Detection

## Run the following Commands To run project on Local Machine

### Step 1: Establish a Conda Environment Upon Accessing the Repository
```bash
conda create -p venv python=3.11 -y
```
### Step 2: Invoke the Conda Environment with the Following Code
```bash
conda activate venv/
```
### STEP 03: Install the necessary requirements using the following command.
```bash
pip install -r requirements.txt
```
### STEP 04: Execute your application within this command prompt.
```bash
streamlit run app.py
```
# Run the following command to create a separate Streamlit application for blur detection
```bash
streamlit run main.py
```
## AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

## Use the following command in production to install OpenCV Python
  ```bash
sudo apt-get install python3-opencv
```

Enjoy Coding!
