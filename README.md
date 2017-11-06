# dont_get_kicked

An old Kaggle challenge which aims to target bad puchasing of an under-risk automobile. 

This problem has imbalanced classes such that in the inital attempt for the problem is to use a boosting method with decision trees as base classifiers. Here in particular, I used AdaBoost with the tree stum. 

Evaluation metric for this problem is set to be ROC-AUC as it is more robust than accuracy for imbalanced data. 

I have included the first stage data exploration, data processing and model in an interative Ipython notebook. Some helper function is wrapped in a separate packege and be loaded in the notebook for use. 

Futher thoughts: 

a)more data processing: oversampling/ downsampling

b) XGBoost


### update:
Added a new ipython notebook which added an exploration using XGBoost on the dataset at the end. The roc_auc score has a slight increase from 0.755 to 0.767 on the validation set.
