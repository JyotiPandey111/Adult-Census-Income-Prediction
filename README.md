conda# Adult-Census-Income-Prediction
Problem Statement: The Goal is to predict whether a person has an income of more than 50K a year or not. This is basically a binary classification problem where a person is classified into the  >50K group or &lt;=50K group.

Approach: The classical machine learning tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing. Try out different machine learning algorithms that’s best fit for the above case.


## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions
5. Terraform


## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections
Link to the Dataset: (https://www.kaggle.com/datasets/overload10/adult-census-dataset)
![image](https://github.com/JyotiPandey111/Adult-Census-Income-Prediction/blob/main/Flowcharts/Data%20Pipeline%20MongoDB.png)

## Project Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)

### Step 1: Clone the repository
```bash
git clone https://github.com/JyotiPandey111/Adult-Census-Income-Prediction.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.8 -y
```

```bash
conda activate venv
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```
