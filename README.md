# dont_get_kicked

An old Kaggle challenge which aims to target bad puchasing of an under-risk automobile. 

### update:

## 03/12/18 Revisit
Added a new notebook with further exploration on feature engineering. 
  - First Attempt to constraint some categorical features to a certain limit (too many categories will lead to overfitting)
  - Second Attempt to treat some object data as text data, and vecotrize them using CountVectorizer and TfidfVectorizer
  
  By fine-tuning both approaches, they can help achieve a learderboard score of 0.23797 (** Note that the 1st place has a score of 0.26719). Much better than the primary attempt wich has dropped a lot of columns during the cardinalities of many collumns are so large.
  
  Change to evaluation metric to F1-score for the imbalanced class issue. 
  
Futher thoughts: 

a)more data processing: oversampling/ downsampling

__b) Neural network on name entity embedding rather using categorical or text features make from vectorizers__

## ~01/2018

Added a new ipython notebook which added an exploration using XGBoost on the dataset at the end. The roc_auc score has a slight increase from 0.755 to 0.767 on the validation set.


This problem has imbalanced classes such that in the inital attempt for the problem is to use a boosting method with decision trees as base classifiers. Here in particular, I used AdaBoost with the tree stum. 

Evaluation metric for this problem is set to be ROC-AUC as it is more robust than accuracy for imbalanced data. 

I have included the first stage data exploration, data processing and model in an interative Ipython notebook. Some helper function is wrapped in a separate packege and be loaded in the notebook for use. 
