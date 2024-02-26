# SelectKBest_dask
This code computes the feature selection for a very large dataset of features.
It execute the feature selection using SelectKBest from sklearn.feature_selection with the scoring function of f_regression for regression tasks  and  dask.delayed function on features (X) and endpoints (y) datasets .
First, load the features csv file (X) and endpoints csv file (y) into pandas dataframe.
Next, perform for loop to iterate through the features and divide them to batches. 
Run the feature_selection function on each batches which uses SelectKBest with f_regression in dask.delayed.
Compute the result of selected features of each batch. 
Concatenate the results of selected features of each batch along columns.
Finally, write the selected features back to csv file.  


