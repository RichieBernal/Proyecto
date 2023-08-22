import logging
import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import ModelPredictor
from api.models.models import Fire

logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.DEBUG) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('FAE/api/main_api.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo


app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("FAE classifier is all ready to go!")
    return 'FAE classifier is all ready to go!'

@app.post('/predict')
def extract_name(fire_features: Fire):
    predictor = ModelPredictor('FAE/models/logistic_regression_output.pkl')
    X = [fire_features.SIZE,
         fire_features.FUEL_lpg,
         fire_features.FUEL_kerosene,
         fire_features.FUEL_thinner,
         fire_features.DISTANCE,
         fire_features.DESIBEL,
         fire_features.AIRFLOW,
         fire_features.FREQUENCY]
    print(f"Input values: {[X]}")
    logger.info(f"Input values:{[X]}")
    prediction = predictor.predict([X])
    logger.info(f"Resultado predicción: {prediction}")
    return JSONResponse(f"Resultado predicción: {prediction}")
