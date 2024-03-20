# -*- coding: utf-8 -*-


import os
import pandas as pd
from sklearn.feature_selection import  SelectKBest, f_regression
from dask.distributed import Client , LocalCluster
import dask

#This code computes the feature selection for a very large dataset of features.
#It execute the feature selection using SelectKBest from sklearn.feature_selection with the scoring function of f_regression for regression tasks  and  dask.delayed function on features (X) and endpoints (y) datasets.
#First, load the features csv file (X) and endpoints csv file (y) into pandas dataframe.
#Next, perform for loop to iterate through the features and divide them to batches. 
# Run the feature_selection function on each batches which uses SelectKBest with f_regression in dask.delayed.
# Compute the result of selected features of each batch .
# Concatenate the results of selected features of each batch along columns.
#Finally, write the selected features back to csv file.  
# Return the number of features in dasaset.

def get_num_descriptors (X_file_path):
    try:
        # Load data from the CSV file which include the header into pandas dataframe
        df = pd.read_csv(X_file_path, sep=',')
        # df_100 = df.iloc [:, :100] 
        
        num_des = df.shape[1]   
        print ( "num_des ", num_des)
    
        return num_des

    except Exception as e:
        print("Error occurred while loading features CSV data:", e)
    
    

# Load data from X_file_path csv file and return pandas dataframe 
def load_X_data(X_file_path):
    
    try:
        # Load data from the CSV file including the header into pandas dataframe
        df = pd.read_csv(X_file_path, sep=',')
        print ( "X dataset  shape ", df.shape)
      

        return df

    except Exception as e:
        print("Error occurred while loading descriptors CSV data:", e)
                
        
# Load data from y_file_path csv file and return pandas dataframe      
def load_y_data(y_file_path):
    
    try:
       
        # Load data from the CSV file including the header into pandas dataframe 
        df = pd.read_csv(y_file_path, sep= ','    )
        print ( "y dataframe shape", df.shape)
        print ( "y dataframe", df)
        
        # exclude the first column of samples' names
        df_modified = df.iloc[:, 1:]
      
        print ( "y sliced dataframe shape", df_modified.shape)
        print ( "y sliced dataframe", df_modified)
        
        return  df_modified

    except Exception as e:
        print("Error occurred while loading label CSV data:", e)
        

   
        
def feature_selection(X_file_path, y_file_path, batch_size, num_selected_features ): 
    
    @dask.delayed
    def kbest_batch( X, y , num_features ):

        # Initialize SelectKBest with the scoring function of f_regression for regression tasks      
        selector = SelectKBest(score_func=f_regression, k = num_features)

        # Fit the selector to your data
        selector.fit(X, y)

        # Get the selected features return boolean
        selected_features = selector.get_support(indices=True)

        selected_feature_names = X.columns[selected_features]
        
        # Retrieve the selected features from the input with their column names
        selected_features = X[selected_feature_names]
        
        # Select the columns corresponding to the selected features
        selected_features = X[selected_feature_names]
        
        return selected_features   
   
    
    X = load_X_data(X_file_path)
    print("X type:", type(X))
    print("X shape:", X.shape)
    
    y = load_y_data(y_file_path) 
    print("y type:", type(y))
    print("y shape:", y.shape)
    print("y contain:", y)
    
    X = X.astype('float32') # Features MUST be float32
    y = y.astype('float32')  # Labels MUST be float32
    
    num_descriptors = get_num_descriptors (X_file_path)   
    
    selected_features = []
    
    # Process the matrix in batches in iterations
    for start in range(0, num_descriptors, batch_size):
        end = min(start + batch_size, num_descriptors)
        
        # Extract the batch of columns
        batch = X.iloc[:, start: end] 
        selected_features.append(kbest_batch (batch, y , num_selected_features))
  
    # Compute the result   
    computed_results = dask.compute(*selected_features)
    print ("computed_results type: ", type(computed_results))   
    print ("computed_results : ", computed_results) 
    
    # Concatenate the results along axis 1
    selected = pd.concat(computed_results, axis=1 )
    print ("selected type: ", type(selected))   
    print ("selected : ", selected) 
    

    
    return selected



# # Function gets the Pandas DataFrame as inpu and  write  Pandas DataFrame into csv and returns file path dictionaty    
def write_to_csv(X_selected, output_path):
    
    print ("X_selected  type: ", type(X_selected))   
    print ("X_selected  shape : ", X_selected.shape) 
     
    
    # Create a separate directory for the output file
    try:
        
      # Create the output directory if it doesn't exist                                                        
       os.makedirs(output_path, exist_ok = True)     
       file_name = 'SelectKBest.csv'
       file_path = os.path.join(output_path, file_name)
                
       X_selected.to_csv(file_path, sep = ',', header = True, index = True ) 

       file_path_dict = {'SelectKBest': file_path}
       print("CSV file written successfully.")
       print ("CSV file size is  " , os.path.getsize(file_path))
       print ("CSV file column number is  " , X_selected.shape[1])
       print ("file_path_dictionary is  " , file_path_dict)
       return file_path 
   
    except Exception as e:
       print("Error occurred while writing matrices to CSV:", e)
       
       
if __name__ == '__main__': 
                  
    
    # Create Lucalluster with specification for each dask worker to create dask scheduler     
    cluster = LocalCluster (n_workers=4,  threads_per_worker=128, memory_limit='500GB',  timeout= 3000)   

    #Create the Client using the cluster
    client = Client(cluster) 

       
    X_file_path = r'/features.csv' 
    y_file_path = r'/endpointcsv'       
    output_path = r'output'
       
    batch_size = 1000
    num_feature_to_select =  100
    
    X_selected = feature_selection (X_file_path, y_file_path, batch_size, num_feature_to_select )
   
    file_path = write_to_csv(X_selected, output_path)
    print (file_path)
 
    scheduler_address = client.scheduler.address
    print("Scheduler Address:", scheduler_address)
    print(cluster.workers)
    print(cluster.scheduler)
    print(cluster.dashboard_link)
    print(cluster.status)
   
    client.close()
    cluster.close()
    


