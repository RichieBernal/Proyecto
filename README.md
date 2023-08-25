# ITESM MLOPs Project

## Introduction of the project

Welcome to the final project focused on MLOps, where the key concepts of ML frameworks and their application will be applied in a practical approach. Throughout this project, the basic concepts and fundamental tools for developing software in the field of MLOps are shown, covering everything from configuring the environment to best practices for creating ML models and deploying them.

## About the project

The overall goal of this project is to build a robust and reproducible MLOps workflow for developing, training, and deploying machine learning models. A linear regression model will be used as a proof of concept due to its simplicity, and it will be applied to the Titanic data set to predict the probability of survival of a passenger based on certain characteristics.

This project covers the following topics:

1. **Key concepts of ML systems**  
The objective of this module is to give an introduction to MLOps, life cycle and architecture examples is also given.

2. **Basic concepts and tools for software development**  
This module focuses on introducing the principles of software development that will be used in MLOps. Consider the configuration of the environment, tools to use, and best practices, among other things.

3. **Development of ML models**  
This module consists of showing the development of an ML model from experimentation in notebooks, and subsequent code refactoring, to the generation of an API to serve the model.

4. **Deployment of ML models**  
The objective of this module is to show how a model is served as a web service to make predictions.

5. **Integration of concepts**  
This module integrates all the knowledge learned in the previous modules. A demo of Continuous Delivery is implemented.

### Baseline

This MLOps project is focused on demonstrating the implementation of a complete workflow that ranges from data preparation to exposing a local web service to make predictions using a linear regression model. The chosen dataset is Acoustic Extinguisher Fire Dataset from Kaggle, which contains information about Classification of Flame Extinction Based on Acoustic Oscillations.

The purpose is to establish a starting point or "baseline" that will serve as a reference to evaluate future improvements and not only more complex algorithms but more complex components and further deployments.

### Scope

This project is planned to cover the topics seen in the course syllabus, which was designed to include technical capacity levels 0, 1 and a small part of 2 of [Machine Learning operations maturity model - Azure Architecture Center | Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

In other words, knowledge is integrated regarding the learning of good software development practices and Dev Ops (Continuous Integration) applied to the deployment of ML models.

### Links to experiments like notebooks

You can find the Acoustic Extinguisher Fire experiments in [folder](/docs/notebooks):

* [1_exploring_data.ipynb](docs\notebooks\1_exploring_data.ipynb)
* [2_general_ideas.ipynb](docs\notebooks\2_general_ideas.ipynb)
* [3_create_convenient_classes.ipynb](docs\notebooks\3_create_convenient_classes.ipynb)
* [4_creating_pipeline.ipynb](docs\notebooks\4_creating_pipeline.ipynb)
* [5_refactored_fae_notebook.ipynb](docs\notebooks\5_refactored_fae_notebook.ipynb)

## Setup

### Python version and packages to install

* Change the directory to the root folder.

* Create a virtual environment with Python 3.10+:

    ```bash
    python3.10 -m venv venv
    ```

* Activate the virtual environment

    ```bash
    source venv/bin/activate
    ```

* Install libraries
Run the following command to install the libraries/packages.

    ```bash
    pip install -r requirements.txt
    ```

## Model training from a main file

To train the Logistic Model, only run the following code:

```bash
python FAE/fae.py
```

Output:

```bash
Directory './data/' already exists.
Index(['SIZE', 'FUEL', 'FUEL_lpg', 'FUEL_kerosene', 'FUEL_thinner', 'DISTANCE',
       'DESIBEL', 'AIRFLOW', 'FREQUENCY'],
      dtype='object')
Index(['SIZE', 'FUEL', 'FUEL_lpg', 'FUEL_kerosene', 'FUEL_thinner', 'DISTANCE',
       'DESIBEL', 'AIRFLOW', 'FREQUENCY'],
      dtype='object')
test roc-auc : 0.9547786269375567
test accuracy: 0.8770421324161651
Model saved in ./models/logistic_regression_output.pkl
```

## Execution of unit tests (Pytest)

### Test location

You can find the test location in the [test](tests) folder, and the following tests:

* Test `test_one_hot_encoder_transform`:  
Test the `transform` method of the OneHotEncoder transformer.

* Test `test_csv_file_existence`:  
Test case to check if the CSV file exists.

* Test `test_model_existence`:  
Test to validate the existence of a `.pkl` model file.

### Execution instructions

#### Test `Data Retriever` class

The following test validates the [load_data.py](FAE\load\load_data.py) module, with the `DataRetriever` class.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_fae.py::test_csv_file_existence -v
    ```

* You should see the following data output:

    ```                                                                         
    tests/test_fae.py::test_csv_file_existence 
    PASSED                                   [100%]

    ```

#### Test model existence

The following test validates the model's existence after the training.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_fae.py::test_model_existence -v -s
    ```

* You should see the following data output:

    ```
    tests/test_fae.py::test_model_existence FAE/models/logistic_regression_output.pkl
    PASSED
    ```

## Usage

### Individual Fastapi and Use Deployment

* Run the next command to start the Titanic API locally

    ```bash
    uvicorn FAE.api.main:app --reload
    ```

#### Checking endpoints

1. Access `http://127.0.0.1:8000/`, you will see a message like this `""FAE classifier is all ready to go!""`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
![FastAPI Docs](docs\img\fast-api.png)
3. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
         "SIZE": 0,
         "FUEL_lpg": 0,
         "FUEL_kerosene": 0,
         "FUEL_thinner": 0,
         "DISTANCE": 0,
         "DESIBEL": 0,
         "AIRFLOW": 0,
         "FREQUENCY": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [1]"
        ```

    * **Prediction 2**  
        Request body

        ```bash
         {
         "SIZE": 6,
         "FUEL_lpg": 0,
         "FUEL_kerosene": 0,
         "FUEL_thinner": 0,
         "DISTANCE": 0,
         "DESIBEL": 0,
         "AIRFLOW": 0,
         "FREQUENCY": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

### Individual deployment of the API with Docker and usage

#### Build the image

* Ensure you are in the `PROYECTO/` directory (root folder).
* Run the following code to build the image:

    ```bash
    docker build -t fae-image ./FAE/app/
    ```

* Inspect the image created by running this command:

    ```bash
    docker images
    ```

    Output:

    ```bash
    REPOSITORY               TAG       IMAGE ID       CREATED              SIZE
    fae-image                latest    540f4e683c3a   About a minute ago   495MB
    ```

#### Run FAE REST API

1. Run the next command to start the `fae-image` image in a container.

    ```bash
    docker run -d --rm --name fae-c -p 8000:8000 fae-image
    docker run -d --rm --name frontend-c -p 3000:5000 frontend-img
    ```

2. Check the container running.

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
    CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS                    NAMES
    2f1f34498517   fae-image   "uvicorn main:app --…"   17 seconds ago   Up 16 seconds   0.0.0.0:8000->8000/tcp   fae-c
    ```

#### Checking endpoints

1. Access `http://127.0.0.1:8000/`, and you will see a message like this `"FAE  classifier is all ready to go!"`
2. A file called `main_api.log` will be created automatically inside the container. We will inspect it below.
3. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
    ![FastAPI Docs](docs\img\fast-api.png)

4. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
        "SIZE": 0,
         "FUEL_lpg": 0,
         "FUEL_kerosene": 0,
         "FUEL_thinner": 0,
         "DISTANCE": 0,
         "DESIBEL": 0,
         "AIRFLOW": 0,
         "FREQUENCY": 0
        }
        ```
        Response body
        The output will be:

        ```bash
        "Resultado predicción: [1]"
        ```

        ![Prediction 1](docs\img\prediction1.png)

    * **Prediction 2**  
        Request body

        ```bash
         {
        "SIZE": 6,
        "FUEL_lpg": 0,
        "FUEL_kerosene": 0,
        "FUEL_thinner": 0,
        "DISTANCE": 0,
        "DESIBEL": 0,
        "AIRFLOW": 0,
        "FREQUENCY": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

        ![Prediction 2](docs\img\prediction2.png)

#### Opening the logs

1. Run the command

    ```bash
    docker exec -it fae-c bash
    ```

    Output:

    ```bash
    root@2f1f34498517:/# 
    ```

2. Check the existing files:

    ```bash
    ls
    ```

    Output:

    ```bash
    Dockerfile   bin   etc   main.py       ml_models  opt        requirements.txt  sbin  tmp README.md    boot  home  main_api.log  mnt    predictor  root   srv   usr __pycache__  dev   lib   media         models     proc       run     sys   var
    ```

3. Open the file `main_api.log` and inspect the logs with this command:

    ```bash
    vim main_api.log
    ```

    Output:

    ```log
    2023-08-23 06:17:23,008:main:main:INFO:FAE classifier is all ready to go!
    2023-08-23 06:30:51,459:main:main:INFO:Input values:[[3, 0, 1, 0, 100, 104, 8.8, 45]]
    2023-08-23 06:30:51,460:main:main:INFO:Resultado predicción: [0]
    2023-08-23 06:34:41,828:main:main:INFO:Input values:[[7, 1, 0, 0, 70, 105, 11.6, 24]]
    2023-08-23 06:34:41,828:main:main:INFO:Resultado predicción: [0]

    ```

4. Copy the logs to the root folder:

    ```bash
    docker cp fae-c:/main_api.log .
    ```

    Output:

    ```bash
    Successfully copied 4.1kB to C:\Users\rbernal\Documents\GitHub\Proyecto\.
    ```

#### Delete container and image

* Stop the container:

    ```bash
    docker stop fae-c
    ```

* Verify it was deleted

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
    CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
    ```

* Delete the image

    ```bash
    docker rmi fae-image
    ```

    Output:

    ```bash
    Deleted: sha256:bb48551cf5423bad83617ad54a8194501aebbc8f3ebb767de62862100d4e7fd2
    ```

### Complete deployment of all containers with Docker Compose and usage

#### Create the network

First, create the network AIService by running this command:

```bash
docker network create AIservice
```

#### Run Docker Compose

* Ensure you are in the directory where the docker-compose.yml file is located

* Run the next command to start the App and Frontend APIs

    ```bash
    docker-compose -f FAE/docker-compose.yml up --build
    ```

    You will see something like this:

    ```bash
    ✔ Container fae-app-1       Created                                                        0.0s 
    ✔ Container fae-frontend-1  Created                                                        0.0s 
    Attaching to fae-app-1, fae-frontend-1
    fae-app-1       | INFO:     Will watch for changes in these directories: ['/']
    fae-app-1       | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    fae-app-1       | INFO:     Started reloader process [1] using StatReload
    fae-frontend-1  | INFO:     Will watch for changes in these directories: ['/']
    fae-frontend-1  | INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    fae-frontend-1  | INFO:     Started reloader process [1] using StatReload
    fae-app-1       | INFO:     Started server process [8]
    fae-app-1       | INFO:     Waiting for application startup.
    fae-app-1       | INFO:     Application startup complete.
    fae-frontend-1  | INFO:     Started server process [8]
    fae-frontend-1  | INFO:     Waiting for application startup.
    fae-frontend-1  | INFO:     Application startup complete.
    ```

#### Checking endpoints in Frontend

1. Access `http://127.0.0.1:3000/`, and you will see a message like this `"Front-end is all ready to go!"`
2. A file called `frontend.log` will be created automatically inside the container. We will inspect it below.
3. Access `http://127.0.0.1:3000/docs`, the browser will display something like this:
    ![Frontend Docs](docs\img\frontend-1.png)

4. Try running the following predictions with the endpoint `classify` by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
        "SIZE": 0,
        "FUEL_lpg": 0,
        "FUEL_kerosene": 0,
        "FUEL_thinner": 0,
        "DISTANCE": 0,
        "DESIBEL": 0,
        "AIRFLOW": 0,
        "FREQUENCY": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [1]"
        ```

        ![Frontend Prediction 1](docs\img\prediction1.png)

    * **Prediction 2**  
        Request body

        ```bash
         {
        "SIZE": 6,
        "FUEL_lpg": 0,
        "FUEL_kerosene": 0,
        "FUEL_thinner": 0,
        "DISTANCE": 0,
        "DESIBEL": 0,
        "AIRFLOW": 0,
        "FREQUENCY": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

        ![Frontend Prediction 2](docs\img\prediction2.png)

#### Opening the logs in Frontend

Open a new terminal, and execute the following commands:

1. Copy the `frontend` logs to the root folder:

    ```bash
    docker cp itesm_mlops_project-frontend-1:/frontend.log .
    ```

    Output:

    ```bash
    Successfully copied 3.07kB to ...\Proyecto\.
    ```

2. You can inspect the logs and see something similar to this:

    ```bash
    INFO: 2023-08-24 07:29:24,371|main|Front-end is all ready to go!
    DEBUG: 2023-08-24 07:30:52,453|main|Incoming input in the front end: {'SIZE': 0, 'FUEL_lpg': 0, 'FUEL_kerosene': 0, 'FUEL_thinner': 0, 'DISTANCE': 0, 'DESIBEL': 0, 'AIRFLOW': 0, 'FREQUENCY': 0}
    DEBUG: 2023-08-24 07:30:53,098|main|Prediction: "Resultado predicción: [1]"
    DEBUG: 2023-08-24 07:31:57,928|main|Incoming input in the front end: {'SIZE': 6, 'FUEL_lpg': 0, 'FUEL_kerosene': 0, 'FUEL_thinner': 0, 'DISTANCE': 0, 'DESIBEL': 0, 'AIRFLOW': 0, 'FREQUENCY': 0}
    DEBUG: 2023-08-24 07:31:57,934|main|Prediction: "Resultado predicción: [0]"
    ```

#### Opening the logs in App

Open a new terminal, and execute the following commands:

1. Copy the `app` logs to the root folder:

    ```bash
    docker cp fae-app-1:/main_api.log .
    ```

    Output:

    ```bash
    Successfully copied 2.05kB to ...\Proyecto\.
    ```

2. You can inspect the logs and see something similar to this:

    ```bash
    2023-08-24 07:30:53,096:main:main:INFO:Input values:[[0, 0, 0, 0, 0, 0, 0.0, 0]]
    2023-08-24 07:30:53,097:main:main:INFO:Resultado predicción: [1]
    2023-08-24 07:31:57,931:main:main:INFO:Input values:[[6, 0, 0, 0, 0, 0, 0.0, 0]]
    2023-08-24 07:31:57,932:main:main:INFO:Resultado predicción: [0]
    ```

### Delete the containers with Docker Compose

1. Stop the containers that have previously been launched with `docker-compose up`.

    ```bash
    docker-compose -f fae/docker-compose.yml stop 
    ```

    Output:

    ```bash
    [+] Stopping 2/2
    ✔ Container fae-frontend-1  Stopped                                                       0.7s 
    ✔ Container fae-app-1       Stopped                                                       1.1s 
    ```

2. Delete the containers stopped from the stage.

    ```bash
    docker-compose -f fae/docker-compose.yml rm
    ```

    Output:

    ```bash
    ? Going to remove fae-frontend-1, fae-app-1 Yes
    [+] Removing 2/0
     ✔ Container fae-app-1       Removed                                              0.0s 
     ✔ Container fae-frontend-1  Removed                                              0.0s 
    ```

## Resources

Here you will find information about this project and more.

### Information sources

* [MNA - Master in Applied Artificial Intelligence](https://learn.maestriasydiplomados.tec.mx/pos-programa-mna-v-)
* [ITESM MLOps Course GitHub Repository](https://github.com/carloslme/itesm-mlops)
* [Google Bard: an artificial intelligence chatbot] (https://bard.google.com/)
* [Phind: AI Search Engine and Pair Programmer] (https://www.phind.com/)

## Contact information

* **Credits**

    ------------

  * **Development Lead**
    * Ricardo Bernal  <rickybernal@gmail.com>
    * [GitHub Profile](https://github.com/RichieBernal)
    * [LinkedIn](www.linkedin.com/in/act-ricardo-bernal)


* **Contributors**
    * Carlos Mejia <carloslmescom@gmail.com>
    * [GitHub Profile](https://github.com/carloslme/)
    * [LinkedIn](https://www.linkedin.com/in/carloslme/)
------------

