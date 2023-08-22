from pydantic import BaseModel


class Fire(BaseModel):
    """
    Represents a passenger on the Fire with various attributes.
    
    Attributes:
        SIZE (int): Placeholder for values in 'SIZE' attribute
        FUEL_lpg (int): Placeholder for values in 'FUEL_lpg' attribute
        FUEL_kerosene (int): Placeholder for values in 'FUEL_kerosene' attribute
        FUEL_thinner (int): Placeholder for values in 'FUEL_thinner' attribute
        DISTANCE (int): Placeholder for values in 'DISTANCE' attribute
        DESIBEL (int): Placeholder for values in 'DESIBEL' attribute
        AIRFLOW (float): Placeholder for values in 'AIRFLOW' attribute
        FREQUENCY (int): Placeholder for values in 'FREQUCENCY' attribute
    """

    SIZE: int  
    FUEL_lpg: int
    FUEL_kerosene: int
    FUEL_thinner: int
    DISTANCE:  int 
    DESIBEL:  int  
    AIRFLOW:  float
    FREQUENCY:  int  
   
    