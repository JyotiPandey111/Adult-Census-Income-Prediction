from income.entity.artifact_entity import ClassificationMetricArtifact
from income.exception import IncomeException
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
import os,sys

def get_classification_score(y_true,y_pred, y_prob)->ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_auc_score  = roc_auc_score(y_true, y_prob)
        

        classsification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    auc_score= model_auc_score)
        return classsification_metric
    except Exception as e:
        raise IncomeException(e,sys)