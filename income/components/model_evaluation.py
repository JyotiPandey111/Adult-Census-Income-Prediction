
from income.exception import IncomeException
from income.logger import logging
from income.entity.artifact_entity import DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from income.entity.config_entity import ModelEvaluationConfig
import os,sys
from income.ml.metric.classification_metric import get_classification_score
from income.ml.model.estimator import IncomeModel
from income.utils.main_utils import save_object,load_object,write_yaml_file
from income.constant.training_pipeline import TARGET_COLUMN
from income.ml.model.estimator import Impute_Missing_Category, TargetValueMapping, NumericaltoCategoricalMapping,CategoricalFeatureTransformer, Encoding_categorical_features, Correlated_independent_feature
import pandas  as  pd
from income.utils.main_utils import read_yaml_file
from income.constant.training_pipeline import SCHEMA_FILE_PATH, CORR_SCHEMA_FILE_PATH

from income.ml.model.estimator import ModelResolver





class ModelEvaluation:


    def __init__(self,model_eval_config:ModelEvaluationConfig,
                    data_validation_artifact:DataValidationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            self.model_eval_config=model_eval_config
            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self._corr_schema_config = read_yaml_file(CORR_SCHEMA_FILE_PATH)

        except Exception as e:
            raise IncomeException(e,sys)
    


    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:

            # FE needed
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            #valid train and test file dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            df = pd.concat([train_df,test_df])


#---------------------------------------------- TRASFORMATION IN df DATAFRAME--------------------------------------------------------


            # handling missing values   in df "missing " category
            for fea in self._schema_config['columns_nan']:
                df = Impute_Missing_Category(data = df, feature=fea).impute_nan()
            logging.info("Successfully impute missing values in concat df category")



            # df data converting numerical features to categorical features  - capital-loss- capital-gain- hours-per-week- age
            df['age'] = NumericaltoCategoricalMapping(data = df, feature= 'age', bins = [0,25,45,65,float('inf')], labels=['Young','Middle-aged', 'Seniors', 'Old']).numericaltocategorical()
            df['hours-per-week'] = NumericaltoCategoricalMapping(data = df, feature= 'hours-per-week', bins = [0,25,40,60,float('inf')], labels=[ 'part-time', 'full-time','over-time', 'too-much']).numericaltocategorical()
# train
            ###############################################################################data[data['capital-gain']>0]['capital-gain'].median()
            logging.info(f" median is df {df[df['capital-loss']>0]['capital-loss'].median()}")
            df['capital-loss'] = NumericaltoCategoricalMapping(data = df, feature= 'capital-loss', bins = [0,1, df[df['capital-loss']>0]['capital-loss'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital loss: {df['capital-loss'].head(5)} ")

            logging.info(f" median is df {df[df['capital-gain']>0]['capital-gain'].median()} ")
            df['capital-gain'] = NumericaltoCategoricalMapping(data = df, feature= 'capital-gain', bins = [0,1, df[df['capital-gain']>0]['capital-gain'].median(),float('inf')], labels=[ 'none', 'low','high']).numericaltocategorical()
            logging.info(f"capital gain: {df['capital-gain'].head(5)}  ")

            logging.info('Converted numerical to categorical features in df data')


            # After converting into categorical features
            #Initialize the processor with a threshold for filtering rare categories. set threshold to 1 percentge.
            df= CategoricalFeatureTransformer(data = df, threshold = 0.01).process_categorical_features()
            logging.info('category with less than 1% threshold created new category rare in df data')
            logging.info(f'Columns before Encoding are:{df.columns} ')



            # Encoding the categorical features
            #nominal_columns:
                # nominal_one_hot_encoding:  relationship, race, sex, country, 
                # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            for fea  in self._schema_config['nominal_one_hot_encoding']:
                df= Encoding_categorical_features(df=df,feature=fea).nominal_one_hot_encoding()
                logging.info(f'{df.head(5)}')
                logging.info(f"Successfully encoded df '{fea}' by nominal_one_hot_encoding \n")
            

            # nominal_one_hot_encoding_top_x: marital-status,occupation, workclass
            df= Encoding_categorical_features(df=df,feature='marital-status', x=5).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded df 'marital-status' by nominal_one_hot_encoding_top_x  \n")
            df= Encoding_categorical_features(df=df,feature='occupation', x =13).nominal_one_hot_encoding_top_x()
            logging.info(f"Successfully encoded df 'occupation 'by nominal_one_hot_encoding_top_x\n")
            df= Encoding_categorical_features(df=df,feature='workclass', x =7).nominal_one_hot_encoding_top_x()
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
            #df dataframe
            y_true = df[TARGET_COLUMN]
            y_true.replace({' <=50K': 0, ' >50K': 1},inplace=True)
            df.drop(TARGET_COLUMN,axis=1,inplace=True)  
            
            logging.info("df and y_true seperated successfully")
            logging.info(f'Columns after encoding are:{df.columns} ')

            logging.info(f'--------------df INPUT DATAFRAME: {df.head(5)}----------')
            logging.info(f'--------------df TARGET DATAFRAME: {y_true[:5]}----------')


            # ew have to delete same columns which we have deleted from train data

            #df = df.drop(columns=self._schema_config['corr_features_spearman_list'], axis=1)
            #logging.info("corr_features_spearman_list dropped from df data")
            #logging.info(f'Columns after deleting spearmen are:{df.columns} ')

            #corr_features_spearman = Correlated_independent_feature(data=df, threshold=0.25, method = 'spearman').correlation()
            #corr_features_spearman_list = []
            #for ele in corr_features_spearman:
            #    corr_features_spearman_list.append(ele)

            logging.info(f"\nList of Correlated Independent Variable: {self._corr_schema_config['corr_features_spearman_list']}")
            #df = df.drop(columns=self._corr_schema_config['corr_features_spearman_list'], axis=1)
            df = df.drop(columns=self._corr_schema_config['corr_features_spearman_list'], axis=1)
            logging.info("corr_features_spearman_list dropped from df data")
            logging.info(f'Columns after deleting spearmen are:{df.columns} ')



# --------------------------------------------df DATAFRAME TRASFORMATION COMPLETED---------------------------------------------------------------------------------           





            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True


            if model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact


            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)
            
            y_trained_pred = train_model.predict(df)
            y_latest_pred  =latest_model.predict(df)

            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.f1_score-latest_metric.f1_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                #0.02 < 0.03
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)

            model_eval_report = model_evaluation_artifact.__dict__

            #save the report
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise IncomeException(e,sys)

    
    