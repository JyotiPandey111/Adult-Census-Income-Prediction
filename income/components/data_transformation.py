import sys, os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


from income.constant.training_pipeline import TARGET_COLUMN
from income.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)

from income.entity.config_entity import DataTransformationConfig
from income.exception import IncomeException
from income.logger import logging

from income.ml.model.estimator import Impute_Missing_Category, TargetValueMapping, NumericaltoCategoricalMapping,CategoricalFeatureTransformer, Encoding_categorical_features, Correlated_independent_feature
from income.utils.main_utils import read_yaml_file,write_yaml_file
from income.constant.training_pipeline import SCHEMA_FILE_PATH, CORR_SCHEMA_FILE_PATH, TRAINED_FEATURES_PATH
from income.utils.main_utils import save_numpy_array_data, save_object




class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self._corr_schema_config = read_yaml_file(CORR_SCHEMA_FILE_PATH)
            self._trained_features_config = read_yaml_file(TRAINED_FEATURES_PATH)

        except Exception as e:
            raise IncomeException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise IncomeException(e, sys)

   

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        '''
        performing data trasformation
        '''
        try:
            #robust_scaler = RobustScaler()
            scaler = StandardScaler()
            

            # missing values are in categorical features so creating new category 'Missing'
            #simple_imputer = SimpleImputer(strategy="constant", fill_value='Missing')





            preprocessor = Pipeline(
                steps=[
                    #("Imputer", simple_imputer), #replace missing values with zero
                    #("RobustScaler", robust_scaler) #keep every feature in same range and handle outlier
                    # log transformation
                    # any data transformation steps as per requirement
                    ("scaler", scaler) # 
                    ]
            )
            logging.info("data transformer object created successfully")

            
            return preprocessor

        except Exception as e:
            raise IncomeException(e, sys) from e

    

        



    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            logging.info('initiating data transformation')

            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            logging.info("read train_df successfully")

            train_df[TARGET_COLUMN]= train_df[TARGET_COLUMN].replace({' <=50K': 0, ' >50K': 1})
            logging.info(f"traget feature train encoded successfully using TargetValueMapping :  {train_df[TARGET_COLUMN][:5]}")


            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info('read test_df successfully')

            test_df[TARGET_COLUMN]= test_df[TARGET_COLUMN].replace({' <=50K': 0, ' >50K': 1})
            logging.info(f"traget feature test encoded successfully using TargetValueMapping:  {test_df[TARGET_COLUMN][:5]}")


            preprocessor = self.get_data_transformer_object()


#---------------------------------------------- TRASFORMATION IN TRAINING DATAFRAME--------------------------------------------------------


            # handling missing values   in training "missing " category
            for fea in self._schema_config['columns_nan']:
                train_df = Impute_Missing_Category(data = train_df, feature=fea).impute_nan()
            logging.info("Successfully impute missing values in training category")



            # Train data converting numerical features to categorical features  - capital-loss- capital-gain- hours-per-week- age
            train_df['age'] = NumericaltoCategoricalMapping(data = train_df, feature= 'age', bins = [0,25,45,65,float('inf')], labels=['Young','Middle-aged', 'Seniors', 'Old']).numericaltocategorical()
            train_df['hours-per-week'] = NumericaltoCategoricalMapping(data = train_df, feature= 'hours-per-week', bins = [0,25,40,60,float('inf')], labels=[ 'part-time', 'full-time','over-time', 'too-much']).numericaltocategorical()

            ###############################################################################data[data['capital-gain']>0]['capital-gain'].median()
            logging.info(f" median is train {train_df[train_df['capital-loss']>0]['capital-loss'].median()}")
            train_df['capital-loss'] = NumericaltoCategoricalMapping(data = train_df, feature= 'capital-loss', bins = [0,1, train_df[train_df['capital-loss']>0]['capital-loss'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital loss: {train_df['capital-loss'].head(5)} ")

            logging.info(f" median is train {train_df[train_df['capital-gain']>0]['capital-gain'].median()} ")
            train_df['capital-gain'] = NumericaltoCategoricalMapping(data = train_df, feature= 'capital-gain', bins = [0,1, train_df[train_df['capital-gain']>0]['capital-gain'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital gain: {train_df['capital-gain'].head(5)}  ")

            logging.info('Converted numerical to categorical features in train data')


            # After converting into categorical features
            #Initialize the processor with a threshold for filtering rare categories. set threshold to 1 percentge.
            train_df= CategoricalFeatureTransformer(data = train_df, threshold = 0.01).process_categorical_features()
            logging.info('category with less than 1% threshold created new category rare in Train data')
            logging.info(f'Columns before Encoding are:{train_df.columns} ')



            # Encoding the categorical features
            #nominal_columns:
                # nominal_one_hot_encoding:  relationship, race, sex, country, 
                # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            for fea  in self._schema_config['nominal_one_hot_encoding']:
                train_df= Encoding_categorical_features(df=train_df,feature=fea).nominal_one_hot_encoding()
                logging.info(f'{train_df.head(5)}')
                logging.info(f"Successfully encoded train '{fea}' by nominal_one_hot_encoding \n")
            

            # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            train_df= Encoding_categorical_features(df=train_df,feature='marital-status', x=5).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded train 'marital-status' by nominal_one_hot_encoding_top_x  \n")
            train_df= Encoding_categorical_features(df=train_df,feature='occupation', x =13).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded train 'occupation 'by nominal_one_hot_encoding_top_x\n")
            train_df= Encoding_categorical_features(df=train_df,feature='workclass', x =7).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded train 'workclass' by nominal_one_hot_encoding_top_x\n")



            #ordinal_columns:education, education,age, hours-per-week, capital_gain, capital_loss
            dictionary_edu={' Some-college':3, ' Bachelors':3, ' Assoc-acdm':2, ' 11th':2,' Assoc-voc':2, 
                        ' Masters':4, ' HS-grad':2, ' Doctorate':5, ' Prof-school':2,' 10th':2, ' 7th-8th':1, 
                        'Other':1, ' 9th':2, ' 12th':2}
            train_df= Encoding_categorical_features(df = train_df, feature='education',dictionary=dictionary_edu).ordinal_label_encoding()
            logging.info(f"Successfully encoded train 'education' by ordinal_label_encoding\n")

            dictionary_age = {'Middle-aged': 2,'Young':1, 'Seniors':3, 'Old':4}
            train_df= Encoding_categorical_features(df = train_df, feature='age',dictionary=dictionary_age).ordinal_label_encoding()
            logging.info(f"Successfully encoded train 'age' by ordinal_label_encoding\n")

            dictionary_hours = {'over-time':3, 'too-much':4, 'part-time':1, 'full-time':2}
            train_df= Encoding_categorical_features(df = train_df, feature='hours-per-week',dictionary=dictionary_hours).ordinal_label_encoding()
            logging.info(f"Successfully encoded train 'hours-per-week' by ordinal_label_encoding\n")

            dictionary_cap = {'none':0, 'low':1, 'high':2}
            train_df= Encoding_categorical_features(df = train_df, feature='capital-gain',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded train 'capital_gain' by ordinal_label_encoding\n")
            train_df= Encoding_categorical_features(df = train_df, feature='capital-loss',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded train 'capital_loss' by ordinal_label_encoding\n")


#---------------------------------------------------
            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)  
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info("input_feature_train_df and traget_feature_train_df seperated successfully")
            logging.info(f'Columns after encoding are:{input_feature_train_df.columns} ')

            logging.info(f'--------------TRAIN INPUT DATAFRAME: {input_feature_train_df.head(5)}----------')
            logging.info(f'--------------TRAIN TARGET DATAFRAME: {target_feature_train_df[:5]}----------')



            corr_features_spearman = Correlated_independent_feature(data=input_feature_train_df, threshold=0.25, method = 'spearman').correlation()
            corr_features_spearman_list= {'corr_features_spearman_list': list(corr_features_spearman)}
            logging.info(f"\nList of Correlated Independent Variable: {corr_features_spearman_list}")
            #input_feature_train_df = input_feature_train_df.drop(columns=self._schema_config['corr_features_spearman_list'], axis=1)
            input_feature_train_df = input_feature_train_df.drop(columns=corr_features_spearman_list['corr_features_spearman_list'], axis=1)
            write_yaml_file(file_path = CORR_SCHEMA_FILE_PATH ,content= corr_features_spearman_list)
            write_yaml_file(file_path = TRAINED_FEATURES_PATH, content = {"trained_features":list(input_feature_train_df.columns)})


            logging.info("corr_features_spearman_list dropped from training data")
            logging.info(f'Columns after deleting spearmen are:{input_feature_train_df.columns} ')



# --------------------------------------------TRAIN DATAFRAME TRASFORMATION COMPLETED--------------------------------------------------------------------------------------





#----------------------------------------------------TRANSFORMING TEST DATAFRAME---------------------------------------------


             # handling missing values   in test "missing " category
            for fea in self._schema_config['columns_nan']:
                test_df = Impute_Missing_Category(data = test_df, feature=fea).impute_nan()
            logging.info("Successfully impute missing values in test category")



            # test data converting numerical features to categorical features  - capital-loss- capital-gain- hours-per-week- age
            test_df['age'] = NumericaltoCategoricalMapping(data = test_df, feature= 'age', bins = [0,25,45,65,float('inf')], labels=['Young','Middle-aged', 'Seniors', 'Old']).numericaltocategorical()
            test_df['hours-per-week'] = NumericaltoCategoricalMapping(data = test_df, feature= 'hours-per-week', bins = [0,25,40,60,float('inf')], labels=[ 'part-time', 'full-time','over-time', 'too-much']).numericaltocategorical()

            ###############################################################################data[data['capital-gain']>0]['capital-gain'].median()
            logging.info(f" median is test {test_df[test_df['capital-loss']>0]['capital-loss'].median()}")
            test_df['capital-loss'] = NumericaltoCategoricalMapping(data = test_df, feature= 'capital-loss', bins = [0,1, test_df[test_df['capital-loss']>0]['capital-loss'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital loss: {test_df['capital-loss'].head(5)} ")

            logging.info(f" median is test {test_df[test_df['capital-gain']>0]['capital-gain'].median()} ")
            test_df['capital-gain'] = NumericaltoCategoricalMapping(data = test_df, feature= 'capital-gain', bins = [0,1, test_df[test_df['capital-gain']>0]['capital-gain'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital gain: {test_df['capital-gain'].head(5)}  ")

            logging.info('Converted numerical to categorical features in Test data')


            # After converting into categorical features
            #Initialize the processor with a threshold for filtering rare categories. set threshold to 1 percentge.
            test_df= CategoricalFeatureTransformer(data = test_df, threshold = 0.01).process_categorical_features()
            logging.info('category with less than 1% threshold created new category rare in test data')
            logging.info(f'Columns before Encoding are:{test_df.columns} ')



            # Encoding the categorical features
            #nominal_columns:
                # nominal_one_hot_encoding:  relationship, race, sex, country, 
                # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            for fea  in self._schema_config['nominal_one_hot_encoding']:
                test_df= Encoding_categorical_features(df=test_df,feature=fea).nominal_one_hot_encoding()
                logging.info(f'{test_df.head(5)}')
                logging.info(f"Successfully encoded test '{fea}' by nominal_one_hot_encoding \n")
            

            # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            test_df= Encoding_categorical_features(df=test_df,feature='marital-status', x=5).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded test 'marital-status' by nominal_one_hot_encoding_top_x  \n")
            test_df= Encoding_categorical_features(df=test_df,feature='occupation', x =13).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded test 'occupation 'by nominal_one_hot_encoding_top_x\n")
            test_df= Encoding_categorical_features(df=test_df,feature='workclass', x =7).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded test 'workclass' by nominal_one_hot_encoding_top_x\n")



            #ordinal_columns:education, education,age, hours-per-week, capital_gain, capital_loss
            dictionary_edu={' Some-college':3, ' Bachelors':3, ' Assoc-acdm':2, ' 11th':2,' Assoc-voc':2, 
                        ' Masters':4, ' HS-grad':2, ' Doctorate':5, ' Prof-school':2,' 10th':2, ' 7th-8th':1, 
                        'Other':1, ' 9th':2, ' 12th':2}
            test_df= Encoding_categorical_features(df = test_df, feature='education',dictionary=dictionary_edu).ordinal_label_encoding()
            logging.info(f"Successfully encoded test 'education' by ordinal_label_encoding\n")

            dictionary_age = {'Middle-aged': 2,'Young':1, 'Seniors':3, 'Old':4}
            test_df= Encoding_categorical_features(df = test_df, feature='age',dictionary=dictionary_age).ordinal_label_encoding()
            logging.info(f"Successfully encoded test 'age' by ordinal_label_encoding\n")

            dictionary_hours = {'over-time':3, 'too-much':4, 'part-time':1, 'full-time':2}
            test_df= Encoding_categorical_features(df = test_df, feature='hours-per-week',dictionary=dictionary_hours).ordinal_label_encoding()
            logging.info(f"Successfully encoded test 'hours-per-week' by ordinal_label_encoding\n")

            dictionary_cap = {'none':0, 'low':1, 'high':2}
            test_df= Encoding_categorical_features(df = test_df, feature='capital-gain',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded test 'capital_gain' by ordinal_label_encoding\n")
            test_df= Encoding_categorical_features(df = test_df, feature='capital-loss',dictionary=dictionary_cap).ordinal_label_encoding()
            logging.info(f"Successfully encoded test 'capital_loss' by ordinal_label_encoding\n")


#---------------------------------------------------
            #test dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)  
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("input_feature_test_df and traget_feature_test_df seperated successfully")
            logging.info(f'Columns after encoding are:{input_feature_test_df.columns} ')

            logging.info(f'{input_feature_test_df.head(5)}')


            #input_feature_test_df = input_feature_test_df.drop(columns=self._schema_config['corr_features_spearman_list'], axis=1)
            #logging.info("corr_features_spearman_list dropped from test data")
            #logging.info(f'Columns after deleting spearmen are:{input_feature_test_df.columns} ')

            # ew have to delete same columns which we have deleted from train data

            #corr_features_spearman = Correlated_independent_feature(data=input_feature_test_df, threshold=0.25, method = 'spearman').correlation()
            #corr_features_spearman_list = []
            #for ele in corr_features_spearman:
            #    corr_features_spearman_list.append(ele)
            #logging.info(f"\nList of Correlated Independent Variable: {self._corr_schema_config['corr_features_spearman_list']}")
            input_feature_test_df = input_feature_test_df.drop(columns=self._corr_schema_config["corr_features_spearman_list"], axis=1)
            logging.info("corr_features_spearman_list dropped from testing data")
            logging.info(f'Columns after deleting spearmen are:{input_feature_test_df.columns} ')



# --------------------------------------------Test DATAFRAME TRASFORMATION COMPLETED----------------------------------------------------------------

#---------------------------
            



# ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            # SCALING   applying pipeline to train and testing data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            # Randome Forest reqired no Scaling other wise comment below two lines
            transformed_input_train_feature = input_feature_train_df
            transformed_input_test_feature = input_feature_test_df

            # uncomment the beloow to line to scale the data 
            #transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            #transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

           


#???????????????????????????????????????????????????????????????????????????????????????????????????????????????????


            #after preprocessing
            smt = SMOTETomek(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            
            
            # concatenationg data
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact


        except Exception as e:
            raise IncomeException(e, sys) from e