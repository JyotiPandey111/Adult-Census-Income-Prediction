from income.configuration.mongo_db_connection import MongoDBClient
import os,sys
from income.exception import IncomeException
from income.logger import logging
from income.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig



'''#Function for checking is exception and logging  working or not.
def test_exception():
    try:
        logging.info("We are checking logging.py")
    
        x = 1/0
    except Exception as e:
        raise IncomeException(e,sys)'''



if __name__ == '__main__':




    '''
    # Checking DataIngestionConfig is working or not
    training_pipeline_config=TrainingPipelineConfig()
    data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    print(data_ingestion_config.__dict__)'''





    
    '''
    #Checking exception working or not
    
    try:
        test_exception()
    except Exception as e:
        print(e)


    #Checking MOngoDb Connections
    mongodb_client = MongoDBClient()
    print('collection name:', mongodb_client.database.list_collection_names())
    '''