
from income.utils.main_utils import load_numpy_array_data
from income.exception import IncomeException
from income.logger import logging
from income.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from income.entity.config_entity import ModelTrainerConfig
import os,sys

#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from income.ml.metric.classification_metric import get_classification_score

from income.ml.model.estimator import IncomeModel
from income.utils.main_utils import save_object,load_object


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise IncomeException(e,sys)


    def perform_hyper_paramter_tunig(self):...
    # gridsearch cv is required
    

    def train_model(self,x_train,y_train):
        '''
        Pass the best model 
        '''
        try:
            #xgb_clf = XGBClassifier()
            rfc = RandomForestClassifier(criterion='gini',max_depth=19,
                                         max_features='log2',min_samples_leaf= 3,
                                         min_samples_split=3,n_estimators=100)
            
            #xgb_clf.fit(x_train,y_train)
            rfc.fit(x_train,y_train)

            #return xgb_clf
            return rfc
        
        except Exception as e:
            raise e
    


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(x_train, y_train)


            y_train_pred = model.predict(x_train)
            #predict probability and keep probabilities for the positive outcome only
            y_train_prob = model.predict_proba(x_train)[:,1]
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred, y_prob=y_train_prob)
            

            # expected_accuracy set to 0.60
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            


            y_test_pred = model.predict(x_test)
            #predict probability and keep probabilities for the positive outcome only
            y_test_prob = model.predict_proba(x_test)[:,1]

            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred,y_prob=y_test_prob)




            #Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")
            

            #saving model
            # 
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            project_model = IncomeModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=project_model)

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise IncomeException(e,sys)