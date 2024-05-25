import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from income.configuration.mongo_db_connection import MongoDBClient
from income.constant.database import DATABASE_NAME
from income.exception import IncomeException
# print

class IncomeData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        creating mongodb client object to make connection
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise IncomeException(e, sys)


    def save_csv_file(self,file_path ,collection_name: str, database_name: Optional[str] = None):
        '''
        save data from mongodb in csv file format
        '''
        try:
            data_frame=pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise IncomeException(e, sys)


    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            take care of any in correct values in dataset--> np.nan
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            
            # if there is any missing data replace it with np.nan

            #df.replace({"na": np.nan}, inplace=True)
            df.replace({" ?": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise IncomeException(e, sys)