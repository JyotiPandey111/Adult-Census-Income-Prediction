from income.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME,TARGET_COLUMN
from income.utils.main_utils import read_yaml_file
from income.constant.training_pipeline import SCHEMA_FILE_PATH
import os
import pandas as pd
import numpy as np
from income.logger import logging
from imblearn.combine import SMOTETomek




class Impute_Missing_Category:
    '''
    Replacing nan value by Missing new category
    Parameters:
        data (pandas DataFrame): The DataFrame containing the data.
        feature (str): The name of the numerical feature to convert.

    Return: Modified Dataframe

    '''
    def __init__(self,data,feature:str):
        self.data = data
        self.feature = feature

    def impute_nan(self):
        """
        This function is used to impute nan values by new category 'Missing' 

        Return: Modified Dataframe
        """
        self.data[self.feature]=np.where(self.data[self.feature].isnull(),"Missing",self.data[self.feature])
        return self.data
                 



# try to make dict {" <=50": 0, " >50": 1} for adult census income prediction
class TargetValueMapping:
    '''
    encode binary target feature neg:0 pos:1
    '''
    def __init__(self,neg: str,pos: str):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    



class NumericaltoCategoricalMapping:
    '''
    Convert a numerical feature into a categorical feature.
    
    Parameters:
        data (pandas DataFrame): The DataFrame containing the data.
        feature (str): The name of the numerical feature to convert.
        bins (list): The bin edges used for categorization.
        labels (list): The labels corresponding to each bin.
    
    Returns:
        pandas Series: The new categorical feature.
    '''

    def __init__(self,data , feature: str,bins: list, labels: list):
        self.data = data
        self.feature = feature
        self.bins = bins
        self.labels = labels

    def numericaltocategorical(self):
         # Use pd.cut to convert numerical feature into categorical
         return pd.cut(self.data[self.feature], bins=self.bins, labels=self.labels,include_lowest=True)
    
    

class CategoricalFeatureTransformer:
    '''
        Initialize the processor with a threshold for filtering rare categories.
        
        Parameters:
            data (pandas DataFrame): The DataFrame containing the data.
            threshold (float): The threshold for filtering rare categories.

        returns: Modified DataFrame
        '''
    def __init__(self, data, threshold: float):
        
        self.threshold = threshold
        self.data = data
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
    
    def process_categorical_features(self):
        '''
        Process categorical features by filtering rare categories.
            
        Returns:
            pandas DataFrame: The DataFrame with processed categorical features.
        '''

        for col in self._schema_config['categorical_columns']:

            # Grouping by categories and calculating the proportion
            temp = self.data.groupby(col)[TARGET_COLUMN].count() / len(self.data)
            
            # Filter categories with proportion greater than threshold
            temp_df = temp[temp > self.threshold].index
            
            # Replace rare categories with 'Rare_var'
            self.data[col] = np.where(self.data[col].isin(temp_df), self.data[col], 'Rare_var')
        
        return self.data



class Encoding_categorical_features:
    """
    Encoding categorical features

    Parameters:
    df: DataFrame
    features: feature tobe encoded
    x: Top x features
    dictionary: used to encode in Lebel encoding

    Return: DataFrame
    
    """
    def __init__(self, df, feature: str, x: int = 0, dictionary: dict = None):
         self.df = df
         self.feature = feature
         self.x = x
         self.dictionary = dictionary
         self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)


    def nominal_one_hot_encoding(self):
        '''
        Performs One Hot Encoding
        '''
        df_dumm= pd.get_dummies(self.df[self.feature],drop_first=True,prefix=self.feature,dtype=int)
        logging.info(f'Columns for {self.feature} are:{df_dumm.columns}')
        self.df = pd.concat([self.df,df_dumm],axis=1)
        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        #self.df[self.feature]
        #logging.info(f"Using One Hot Encoding for '{self.feature}' column")
        

        return self.df



    def nominal_one_hot_encoding_top_x(self):
        '''
        This function performs a one-hot encoding for features with high cardinality.
        '''
        # function to create the dummy variables for the most frequent labels
        # we can vary the number of most frequent labels that we encode
        # let's make a list with the most frequent categories of the variable

        top_x_labels = [y for y in self.df[self.feature].value_counts().sort_values(ascending=False).head(self.x).index]
        logging.info(f'Top {self.x} most frequent labels in {self.feature} are:{top_x_labels} ')

        for label in top_x_labels:
            self.df[self.feature + '_' + label] = np.where(self.df[self.feature]==label, 1, 0)
            logging.info(f"{self.df[self.feature + '_' + label].head(5)}")

        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"Using One Hot Encoding with Top {self.x} most frequent labels for '{self.feature}' column")
        logging.info(f'Columns after encoding x for {self.feature} are:{self.df.columns}')
        return self.df


    def nominal_frequency_encoding(self):
        '''
        Performs frequency encoding
        '''
        freq=self.df[self.feature].value_counts().to_dict()
        self.df[self.feature + '_freq']=self.df[self.feature].map(freq)
        logging.info(f"{self.df[self.feature + '_freq'].head(5)}")
        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"Using nominal_frequency_encoding for '{self.feature}' column")
        
        return self.df



    def nominal_mean_encoding(self):
        '''
        Performs mean encoding
        '''
        mean_nominal=self.df.groupby([self.feature])[TARGET_COLUMN].mean().to_dict()
        self.df[self.feature +'_mean_encoded']=self.df[self.feature].map(mean_nominal)

        logging.info(f"{self.df[self.feature +'_mean_encoded'].head(5)}")

        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"Performing Mean Encoding on '{self.feature}'column")
        
        return self.df



    # prones to overfitting
    def nominal_prob_ratio(self):
        '''
        Performs probability ration encoding
        disadvantage: prones to overfitting
        '''
        prob_df=self.df.groupby([self.feature])[TARGET_COLUMN].mean()
        #print(prob_df)
        prob_df=pd.DataFrame(prob_df)
        prob_df['not_'+TARGET_COLUMN]=1-prob_df[TARGET_COLUMN]
        # probability of not being a target
        prob_df['Probability_ratio']=prob_df[TARGET_COLUMN]/prob_df['not_'+TARGET_COLUMN]
        probability_encoded=prob_df['Probability_ratio'].to_dict()
        #print(probability_encoded)
        self.df[self.feature+'_encoded']=self.df[self.feature].map(probability_encoded)

        logging.info(f"{self.df[self.feature+'_encoded'].head(5)}")

        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"Performing Probability ration encodeing on '{self.feature}' column")
        logging.info(f'{self.df[self.feature].head(5)}')
        return self.df




    def ordinal_label_encoding(self):
        '''
        Performs Label Encoding
        '''
        self.df[self.feature+'_ordinal']=self.df[self.feature].map(self.dictionary)
        logging.info(f"{self.df[self.feature+'_ordinal'].head(5)}")

        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"Performimng Label Encoding on '{self.feature}' column")
        
        return self.df
    


     # if same frequency of datapoint that is the disadvantage
    def ordinal_target_guided(self):
        '''
        Performs target guided encoding
        disad: same frequency of datapoint
        '''
        ordinal_labels=self.df.groupby([self.feature])[TARGET_COLUMN].mean().sort_values().index
        ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
        self.df[self.feature+'_tar_encoded']=self.df[self.feature].map(ordinal_labels2)
        self.df.drop(axis=1,columns=[self.feature],inplace=True)
        logging.info(f"{self.df[self.feature+'_tar_encoded'].head(5)}")
        #self.df = self.df.drop(axis=1,columns=[variable])
        logging.info(f"Performimng Target Guided Encoding on '{self.feature}' column")
        logging.info(f'{self.df[self.feature].head(5)}')
        return self.df









#Write a code to train model and check the accuracy.

class IncomeModel:

    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e
    


class ModelResolver:
    '''
    Gives us the the best model
    '''

    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e
 
    def get_best_model_path(self,)->str:
        '''
        To get the Latest timestamp model
        '''
        try:
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e

    def is_model_exists(self)->bool:
        '''
        to get the model path which is going to save
        '''
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e
        
