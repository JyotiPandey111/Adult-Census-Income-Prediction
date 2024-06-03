from income.configuration.mongo_db_connection import MongoDBClient
from income.exception import IncomeException
import os,sys
from income.logger import logging
from income.pipeline import training_pipeline
from income.pipeline.training_pipeline import TrainPipeline
import os
from income.utils.main_utils import read_yaml_file
from income.constant.training_pipeline import SAVED_MODEL_DIR
from income.constant.application import APP_HOST, APP_PORT
from income.ml.model.estimator import ModelResolver,TargetValueMapping
from income.utils.main_utils import load_object

from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response, ORJSONResponse
from fastapi import FastAPI, File, Request, UploadFile, Path

from income.pipeline.prediction_pipeline import relationship, education, workclass, occupation, race,sex, maritalstatus
from income.utils.main_utils import read_yaml_file
from income.constant.training_pipeline import SCHEMA_FILE_PATH
import os
import pandas as pd
from income.ml.model.estimator import Impute_Missing_Category, TargetValueMapping, NumericaltoCategoricalMapping,CategoricalFeatureTransformer, Encoding_categorical_features


env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']


# creating instance of FastAPI()
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# decorator for "/" base path
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")




# decorator for train path with path operation get
@app.get("/train")
# defining train path operation function
async def train_route():
    try:

        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    
    except Exception as e:
        return Response(f"Error Occurred! {e}")










# decorator for predict path
#@app.get("/predict")
#async def predict_route(request:Request,file: UploadFile = File(...)):

@app.post("/predict")
async def predict_route( age : int , workclass:workclass , education : education ,maritalstatus : maritalstatus,occupation: occupation, relationship : relationship,  race: race,sex:sex, capgain : int , caploss : int , hours : int  ):  # =  Path(ge= 1, le = 100)
    try:
        #Code to get data from user csv file
        schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        if relationship.value in schema_config['relationship'] and education.value in schema_config['education'] and maritalstatus.value in schema_config['maritalstatus'] and workclass.value in schema_config['workclass'] and occupation.value in schema_config['occupation'] and race.value in schema_config['race'] and sex.value in schema_config['sex']:
            json_data = {
                "age" : age,
                "workclass" : workclass,
                "education" : education,
                "marital-status" : maritalstatus,
                "occupation" : occupation,
                "relationship": relationship,
                "race" : race,
                "sex" : sex,
                "capital-gain" : capgain,
                "capital-loss" : caploss,
                "hours-per-week" : hours
                }
            
            logging.info(f"\n\nDataRecord from FastAPI: {json_data} \n type is: {type(json_data)}\n all keys are : {json_data.keys()}\n items: {json_data.items()} \n race key has value: {json_data['race']} \n education key has value: {json_data['education']}")
        
            dic_record = {k: v for k, v in json_data.items()}

            logging.info(f"\n\nDataRecord after creating new dictionary: {dic_record}")

            #dict2 = {key:value for key, value in json_data.items() if key in required_fields}

            df = pd.DataFrame(data = dic_record)
            logging.info(f"DataRecord: {df}")
            logging.info(f'Columns are:{df.columns} ')

            # FE needed
            #---------------------------------------------- TRASFORMATION IN df DATAFRAME--------------------------------------------------------


            # handling missing values   in df "missing " category
            #for fea in self._schema_config['columns_nan']:
            #    df = Impute_Missing_Category(data = df, feature=fea).impute_nan()
            #logging.info("Successfully impute missing values in concat df category")



            # df data converting numerical features to categorical features  - capital-loss- capital-gain- hours-per-week- age
            df['age'] = NumericaltoCategoricalMapping(data = df, feature= 'age', bins = [0,25,45,65,float('inf')], labels=['Young','Middle-aged', 'Seniors', 'Old']).numericaltocategorical()
            df['hours-per-week'] = NumericaltoCategoricalMapping(data = df, feature= 'hours-per-week', bins = [0,25,40,60,float('inf')], labels=[ 'part-time', 'full-time','over-time', 'too-much']).numericaltocategorical()
# train
            ###############################################################################data[data['capital-gain']>0]['capital-gain'].median()
            
            df['capital-loss'] = NumericaltoCategoricalMapping(data = df, feature= 'capital-loss', bins = [0,1, 1887.0,float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital loss: {df['capital-loss'].head(5)} ")

            
            df['capital-gain'] = NumericaltoCategoricalMapping(data = df, feature= 'capital-gain', bins = [0,1, 7298.0,float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital gain: {df['capital-gain'].head(5)}  ")

            logging.info('Converted numerical to categorical features in df data')


            # After converting into categorical features
            #Initialize the processor with a threshold for filtering rare categories. set threshold to 1 percentge.
            #df= CategoricalFeatureTransformer(data = df, threshold = 0.01).process_categorical_features()
            #logging.info('category with less than 1% threshold created new category rare in df data')
            logging.info(f'Columns before Encoding are:{df.columns} ')



            # Encoding the categorical features
            #nominal_columns:
                # nominal_one_hot_encoding:  relationship, race, sex, country, 
                # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass

            for fea  in schema_config['nominal_one_hot_encoding']:
                df= Encoding_categorical_features(df=df,feature=fea).nominal_one_hot_encoding()
                logging.info(f'{df.head(5)}')
                logging.info(f"Successfully encoded df '{fea}' by nominal_one_hot_encoding \n")
            

            # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            df= Encoding_categorical_features(df=df,feature='marital-status').nominal_one_hot_encoding()
            logging.info(f"Successfully encoded df 'marital-status' by nominal_one_hot_encoding  \n")
            df= Encoding_categorical_features(df=df,feature='occupation').nominal_one_hot_encoding()
            logging.info(f"Successfully encoded df 'occupation 'by nominal_one_hot_encoding_top_x\n")
            df= Encoding_categorical_features(df=df,feature='workclass').nominal_one_hot_encoding()
            logging.info(f"Successfully encoded df 'workclass' by nominal_one_hot_encoding_top_x\n")



            #ordinal_columns:education, education,age, hours-per-week, capital_gain, capital_loss
            dictionary_edu={' Some-college':3, ' Bachelors':3, ' Assoc-acdm':2, ' 5th-6th':1, ' 11th':2,' Assoc-voc':2, 
                        ' Masters':4, ' HS-grad':2, ' Doctorate':5, ' Prof-school':2,' 10th':2, ' 7th-8th':1, 
                        'Rare_var':0, ' 9th':2, ' 12th':2}
            df= Encoding_categorical_features(df = df, feature='education',dictionary=dictionary_edu).ordinal_label_encoding()
            logging.info(f"Successfully encoded df 'education' by ordinal_label_encoding\n")

            dictionary_age = {'Middle-aged': 2,'Young':1, 'Seniors':3, 'Old':4}
            df= Encoding_categorical_features(df = df, feature='age',dictionary=dictionary_age).ordinal_label_encoding()
            logging.info(f"Successfully encoded df 'age' by ordinal_label_encoding\n")

            dictionary_hours = {'over-time':3, 'too-much':4, 'part-time':1, 'full-time':2}
            df= Encoding_categorical_features(df = df, feature='hours-per-week',dictionary=dictionary_hours).ordinal_label_encoding()
            logging.info(f"Successfully encoded df 'hours-per-week' by ordinal_label_encoding\n")

            dictionary_cap = {'none':0, 'low':1, 'high':2}
            df= Encoding_categorical_features(df = df, feature='capital-gain',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded df 'capital_gain' by ordinal_label_encoding\n")
            df= Encoding_categorical_features(df = df, feature='capital-loss',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded df 'capital_loss' by ordinal_label_encoding\n")

#---------------------------------------------------
             
            logging.info(f'Columns after encoding are:{df.columns} ')

            logging.info(f'--------------df INPUT DATAFRAME: {df.head(5)}----------')
            
            df = df.drop(columns=schema_config['corr_features_spearman_list'], axis=1)
            logging.info("corr_features_spearman_list dropped from df data")
            logging.info(f'Columns after deleting spearmen are:{df.columns} ')
# --------------------------------------------df DATAFRAME TRASFORMATION COMPLETED--------------------------------------------------------------------------------- 


            model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
            if not model_resolver.is_model_exists():
                return Response("Model is not available")
        
            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)


            y_pred = model.predict(df)
            df['predicted_column'] = y_pred
            df['predicted_column'].replace({0 : ' <=50K', 1 : ' >50K'},inplace=True)
                    
            #decide how to return file to user.
            if df['predicted_column'] == 0:
                return Response("Income is Less than 50 ")
            else: 
                return Response("Income is greater than 50K")
    
        
        #convert csv file to dataframe
        #df = pd.read_csv(file.file)

    
    except Exception as e:
        raise Response(f"Error Occured! {e}")





def main():
    try:

        #set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
         print(e)
         logging.exception(e)


if __name__=="__main__":
    # comment karo
    main()
    # set_env_variable(env_file_path)

    # uncommnet karo
    app_run(app, host=APP_HOST, port=APP_PORT)
    