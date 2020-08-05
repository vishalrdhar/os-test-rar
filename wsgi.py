#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!python3
#
# Logistic Regression 
#


# In[2]:


#
# Import Libraries
#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Feature Engineering
import feature_engine.categorical_encoders as ce
import feature_engine.missing_data_imputers as mdi
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Model and Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,recall_score

import warnings

import sys

from flask import Flask, render_template, request
from werkzeug.serving import run_simple


# In[3]:


#
# Run environment setup
#

# ignore  warnings
warnings.simplefilter("ignore")

# numpy print options
# used fixe dpoint notation for floats with 4 decimals
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# Display options on terminal for pandas dataframes
pd.options.display.max_columns = None   # None means unlimited
pd.options.display.max_rows = None      # None means unlimited

# global variable is available to all functions in this python file
TRAINED_MODEL = 0
SCALER = 0
MODEL_BUILT = False

# In[4]:


#### Data Exploration


# In[5]:


# Read the data file into a pandas dataframe

def read_data():
    
    print("\n*****FUNCTION read_data\n")
    
    df = pd.read_csv('titanic.csv')
    
    return(df)

# end of function read_data


# In[6]:


def disp_null_counts(df):
    
    print("\n*****FUNCTION disp_null_counts\n")
    
    print("\ndf.shape", df.shape)
    
    # Count Nulls 
    print("\ndataframe null count\n", df.isnull().sum(), sep="")
    
    # Percent of Nulls
    print ("\ndataframe null percentage\n", df.isnull().mean(), sep="")

    return(df.isnull().mean())

# end of function disp_null_counts


# In[7]:


def disp_df_info(df):
    
    print("\n*****FUNCTION disp_df_info\n")
    
    # Full Data set
    print("\ndf.shape\n", df.shape)
    
    # display column labels
    print("\nColumn labels\n", df.columns, sep="")
    
    # display top of data sample list
    # print("\ndf.head\n", df.head(), sep="")
    
    # print first 10 data samples
    print("\ndf.head(10)\n", df.head(10), sep="")
    
    # Identify the Categorical Vars and identify nulls
    print("\ndataframe info\n", df.info(), sep="")
    
    disp_null_counts(df)
    
#     # Count Nulls 
#     print("\ndataframe null count\n", df.isnull().sum())
    
#     # Percent of Nulls
#     print ("\ndataframe null percentage\n", df.isnull().mean())
    
    print("\ndf unique value counts\n", df.nunique(axis=0), sep="")
    print()
    print(df.groupby('Age').size().reset_index(name="Age count"))
#     for col in df.columns:
#         print(df.groupby(col).size().reset_index(name=col+" count"))
    
    # Describe the df
    print ("\ndataframe description\n", df.describe(), sep="")
    print("\ndata features median\n", df.median(axis=0, skipna=True), sep="")
    
# end of function disp_df_info


# In[8]:


#
# function: clean_data
#
# 1. Delete unwanted data features
# 2. Clean up nulls, NaNs and empty cells
#
def clean_data(df_input, cleanoption):
    
    print("\n*****FUNCTION clean_data\n")
    
    df = df_input.copy(deep=True)
    
    # Drop unwanted columns
    # based on visual inspection and domain experience
    # Due to nature of the column values having nothing to do with results
    df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
    print("\n DROPPED columns\n","['PassengerId','Name','Ticket']" )
    #
    # Handle nulls, NANs and empty cells
    #
    # Drop rows with Nulls using df.dropna(), will drop over 20% data NOT OK
    #     df = df.dropna()
    #     print("\nprepared df head and shape\n", df.shape, "\n", df.head())
    #
    if cleanoption == 1:
        # Embarked has 2 nulls, OK to drop rows with a low number of Nulls
        df = df[df['Embarked'].notnull()]
    else:
        # Analyze each column separately
        #
        null_count_percentage = disp_null_counts(df)
        print("\n null_count_percentage -> df.isnull.mean\n", null_count_percentage, "\n", type(null_count_percentage), sep="")

        nullhighthreshold = 0.34
        nulllowthreshold = 0.10
        
        column_list = df.columns.values.tolist()
        # Remove the y column from this clean up
        column_list.remove('Survived')
        
        for col in column_list:
            print("\n NULL COUNT PERCENTAGE for",col, "\n", null_count_percentage.loc[col])
            if null_count_percentage.loc[col] >= nullhighthreshold:
                # delete the column
                df.drop([col], axis="columns", inplace=True)
                print("\n DROPPED columns\n", col )
            elif null_count_percentage.loc[col] >= nulllowthreshold:
                # fix the data
                pass

        # delete the remaining rows with nulls, NaNs, empty cells
        print("\n number of null rows", df.shape[0] - df.dropna(inplace=False).shape[0] )
        print("\n Number of DROPPED ROWS:", df.shape[0] - df.dropna(inplace=False).shape[0])
        df.dropna(axis = 'index', inplace=True)
        
    # end if cleanoption == 1 else for nulls 
    
    
    if cleanoption == 1:
        # Drop unwanted columns
        # based on visual inspection and domain experience
        # Due to high number of unique values
        df.drop(['Cabin'], axis=1, inplace=True)
        print("\n DROPPED columns\n","['Cabin']" )
    elif cleanoption == 2:
        #
        # Test for unique values count >= 70% of total data samples
        #
        unique_value_counts = df.nunique(axis=0)
        numrows = df.shape[0]
        uniquethreshold = int(0.75 * numrows)
        cols_to_drop = []
        for col in df.columns:
            if unique_value_counts.loc[col] >= uniquethreshold:
                cols_to_drop.append(col)
        print("\n**** DROPPED columns Data features (columns) dropped due to >= 75% unique values\n", cols_to_drop)
        df.drop(df[cols_to_drop], axis=1, inplace=True)
    else:
        sys.exit(666)
    
    null_counts = disp_null_counts(df)
    print("\n null_counts -> df.isnull.mean\n", null_counts, "\n", type(null_counts), sep="")
    print("\n  after clean_data, shape =\n", df.shape, sep="")
    
    return(df)

#     df['Cabin'] = df['Cabin'].str[0]
#     print("\nCabin\n", df['Cabin'])

# end of functiom clean_data


# #### Feature Engineering

# In[9]:


def feature_engineering(df_input):
    
    print("\n*****FUNCTION feature_engineering\n")
    
    df = df_input.copy(deep=True)
    
    # Create a Pie Chart to check Balance
    df['Survived'].value_counts(sort=True)
    
    #Plotting Parameters
    plt.figure(figsize=(5,5))
    sizes = df['Survived'].value_counts(sort=True)
    colors = ["grey", 'purple']
    labels = ['No', 'Yes']

    #Plot
    plt.pie(sizes, colors = colors, labels = labels, autopct='%1.1f%%', shadow=True, startangle=270,)

    plt.title('Percentage of Churn in Dataset')
    plt.show()
    
    df['Age'].hist(bins=30)
    plt.show()
    
    df['Fare'].hist(bins=30)
    plt.show()
    
    print("\nBEFORE PIPELINE\n")
    disp_null_counts(df)

    # Set up a Feature Engineering pipleine 

    titanic_pipe = Pipeline([
        # replace NA, NaNs, nulls with median of the non-null cells
        ('median_imputer', mdi.MeanMedianImputer(imputation_method='median',
                                            variables=['Age','Fare'])),             

        ('ohce1',ce.OneHotCategoricalEncoder(variables=['Sex'], 
                                            drop_last=True)),                             

        ('ohce2',ce.OneHotCategoricalEncoder(variables=['Embarked'], 
                                            drop_last=False)),                             

       ])  
    
    ###
    ###
    ### ALTERNATIVE METHOD OF DOING ABOVE OHCE One-Hot Encoding
    ###
#     df['Sex'].replace(['male','female'],[0,1], inplace = True)
#     onehot = pd.get_dummies(df['Embarked'])
#     df = df.drop('Embarked', axis = 'columns')
#     df = df.join(onehot)

#     df['Embarked'].replace(['C','Q','S'],[0,1,2], inplace = True)
    ###
    ###
    ###
    
    print("\nAFTER PIPELINE definition, before pipeline fit&transform\n")
    disp_null_counts(df)
    print("\ndf unique value counts\n", df.nunique(axis=0), sep="")
    print("\n",df.groupby('Age').size().reset_index(name="Age count"), sep="")
    
    # Fit will ONLY learn the mean, median values
    # Transform will Apply the changes to the df
    #
    # use the mean and median from the training data for 
    # transform of the new data for the trained model
    titanic_pipe.fit(df)   
    df = titanic_pipe.transform(df)
    
    print("\nafter pipeline fit&transform\n")
    
    disp_null_counts(df)
    print("\ndf unique value counts\n", df.nunique(axis=0), sep="")
    print("\n", df.groupby('Age').size().reset_index(name="Age count"), sep="")
    
    # Transformed df - No Nulls after imputation
    print("\nNulls after transformation\n", df.isnull().sum(), sep="" )
    
    # Transformed df - dummy vars created
    print("\ndf head after pipeline transform (ohe and median)\n", df.head(), sep = "")

    
    return(df)
    
# end of function feature_engineering


# #### Feature Selection

# In[10]:


def feature_selection(df_input):
    
    print("\n*****FUNCTION feature_selection\n")
    
    df = df_input.copy(deep=True)
    
    # Correlation Matrix visualized as HeatMap
    corr_mat = df.corr()
#     plt.figure(figsize=(8,8))
#     sns.heatmap(corr_mat, annot= True, cmap='coolwarm', center = 0 , vmin=-1, vmax=1)
#     plt.show()
    
    # Create Y var
    y = df['Survived']
#     y.head(10)
    
    # Create X var
    x = df.drop(['Survived'], axis=1)
#     x.head(10)
    
    #https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    # Create correlation matrix
    corr_matrix = x.corr().abs()
    print(corr_matrix)
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print(upper)
    
    # Find index of feature columns with correlation greater than a user set value 
    to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
    print(to_drop)
    
    # Shape before dropping features
    print('Shape BEFORE Dropping features:', x.shape)

    # Drop features
    x.drop(x[to_drop], axis=1, inplace=True)

    # Shape after dropping features
    print('Shape AFTER Dropping features:', x.shape)

    print("head after dropping features\n", x.head())
    
    return(x,y)
    
# end of function feature_selection


# #### Feature Scaling

# In[11]:


def feature_scaling(x_input):
    
    global SCALER
    
    x = x_input.copy(deep=True)
    
    print("\n*****FUNCTION feature_scaling \n")

    # Choose columns to scale ,create a separate df
    colstoscale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    dftoscale = x[colstoscale]
    dftoscale.head()
        
    # Call scaler to scale the df
    SCALER = MinMaxScaler()
    dftoscale = SCALER.fit_transform(dftoscale)
    print(dftoscale)
    
    # put the scaled df back into original df
    x[colstoscale] = dftoscale
    print("\nScaled x values\n", x.head())
    print("\nScaled x values\n", x) 
    
    #### Results of scaling 
    print("\nResults of scaling\n", x.describe())
    
    print("\nmax - min of each column\n")
    print(x.max()- x.min())

    return(x)
    
# end of function feature_scaling


# In[12]:


#
# prepare the new sample data in same way training data was prepared
#
def prep_new_data(df_new_data):
    
    print("\n*****FUNCTION prep_new_data \n")
    
    df = df_new_data.copy(deep=True)
    
    # Drop unwanted columns
    # based on visual inspection and domain experience
    # Due to high number of unique values
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
    
    # discard any sample rows with a null/NaN in any column/data feature
    print("\n Number of DROPPED ROWS:", df.shape[0] - df.dropna(inplace=False).shape[0])
    df.dropna(axis = 'index', inplace=True)
    
    if df.shape[0] == 0:
        return(df)
    
    df['Sex'].replace(['male','female'],[0,1], inplace = True)
    
    df['Embarked_S'] = df['Embarked']
    df['Embarked_C'] = df['Embarked']
    df['Embarked_Q'] = df['Embarked']
    df['Embarked_S'].replace(['S','Q', 'C'],[1,0,0], inplace = True)
    df['Embarked_C'].replace(['S','Q', 'C'],[0,0,1], inplace = True)
    df['Embarked_Q'].replace(['S','Q', 'C'],[0,1,0], inplace = True)
            
    df = df.drop('Embarked', axis = 'columns')
    df = df.drop('Embarked_C', axis = 'columns')
      
    #
    # scale remaining data
    #
    # Choose columns to scale ,create a separate df
    colstoscale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    dftoscale = df[colstoscale]
    
    # Call the scaler used for the training data to 
    # do the scale transformation of the new data sample
    dftoscale = SCALER.transform(dftoscale)
    print(dftoscale)
    
    # put the scaled df back into original df
    df[colstoscale] = dftoscale
    
    return(df)
    

# end of function prep_new_data


# In[13]:


#### Model Fitting 


# In[14]:


def build_logreg_model():
    
    print("\nFUNCTION build_logreg_model\n")

    # call get data
    titanic_df = read_data()

    # display dataframe info
    print("Titanic Dataframe information\n")
    disp_df_info(titanic_df)
    # sys.exit(888)
    
    # Clean up data
    cleanoption = 2
    prep_df = clean_data(titanic_df, cleanoption)
    
    # Build the model
    
    # get x, y data
    # x = prep_df.drop(labels = ['Survived'], axis = 1 )
    # y = prep_df['Survived'].values
    
    # feature engineering
    prep_df = feature_engineering(prep_df)

    # Feature Selection
    # x values will be the data features used in training of the model
    # y values are the known predictions for the x used in training the model
    x,y = feature_selection(prep_df)
    
    # Feature Scaling
    x = feature_scaling(x)
    
    disp_df_info(x)
    
    # Train the model
    # Call Logistic Regession with no penalty
    mod = LogisticRegression(penalty='none')
    mod.fit(x,y)
    # OR mod = LogisticRegression().fit(x,y)
    
    # get model accuracy 
    # Score the model
    score = mod.score(x, y)
    print('Accuracy Score is:',score)

    # probability converted to predictions
    y_pred = mod.predict(x)
    y_pred= pd.DataFrame(y_pred)
    y_pred.head()

#
# Model Metrics
#
    print("\n", 44 * "*", sep="")
    print("************* MODEL METRICS ****************")
    print(44 * "*", sep="")
    # Confusion Matrix gives the mistakes made by the classifier
    print("\nConfusion Matrix")
    cm =confusion_matrix(y, y_pred)
    print(cm)
    
    # Confusion Matrix visualized
    plt.figure(figsize= (8,6))
    sns.heatmap(cm, annot= True, fmt= 'd', cmap = 'Reds')
    plt.xlabel('Predicted y_pred')
    plt.ylabel('Actuals / labels - y')
    plt.show()
    
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # For Logistic Regression the model score is the Accuracy Ratio
    # (TP+TN)/(TP+TN+FP+FN)
    acc = accuracy_score(y,y_pred)
    print('\nAccuracy:',acc)
    
        # Precion = TP/(TP+FP)
    # Interpretation: out of all the predicted positive classes, how much we predicted correctly.
    pre = precision_score(y,y_pred)
    print('Precision:',pre)

    # Specificity = TN/(TN+FN)
    # Interpretation: out of all the -ve samples, how many was the classifier able to pick up
    spec = TN/(TN + FP)

    # Recall/Sensitivity/tpr = TP/(TP+FN)
    # Interpretation: out of all the +ve samples, how many was the classifier able to pick up
    rec = recall_score(y,y_pred)
    tpr=rec
    print('Recall:',rec)

    # false positive rate(fpr) = FP/(FP + TN) = 1-specificity
    # Interpretation: False alarm rate
    fpr = FP/(FP + TN)
    print('False Positive Rate',fpr)
    
    print("\n", 51 * "*", sep="")
    print("************* END OF MODEL METRICS ****************")
    print(51 * "*", sep="")
    # end of model metrics

    # return the trained model
    MODEL_BUILT = True
    return(mod)
    
# end of function build_logreg_model


# In[ ]:





# In[15]:


def make_prediction(x_input):
    
    print("\nFUNCTION make_prediction\n")
    
    the_prediction = TRAINED_MODEL.predict(x_input)
    return (the_prediction)
    
# end of function make_prediction


# In[16]:


def get_new_data():
    
    #
    # quick laundry list of features in one data sample
    # For purposes of example, ASSUME: new data for predictions will be same format as training csv
    #
    xdata =  [ [ "1234567", 3, "RayBob", "male", "32", 3, 2, "456789", 37.05, "B13", "Q" ] ]
    #xdata[0][6] = np.NaN
    
    
    new_x_df = pd.DataFrame(data=xdata, columns = ['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    
    return (new_x_df)
    
# end of function get_new_data


# In[ ]:


# 
# Code Starts Here!
#
if __name__ == "__main__":
    
    print("Python special variable __name__ =", __name__)
    # if __main__, file was main file run standalone
    print ("\nPython script is run standalone\n")
        
    # run the flask app
    application.run(host='0.0.0.0', debug=True)
    
else:
    print("\nPython script was imported")

    # build the model
    # function returns a trained model
    if not MODEL_BUILT:
        TRAINED_MODEL = build_logreg_model()
        
    # instantiate the Flask object and then run it
    application = Flask(__name__)
	
   
    @application.route('/',methods = ['POST', 'GET'])
    def index():
        return render_template('input_template.html')
    # end of function
    
    
    @application.route('/result',methods = ['POST', 'GET'])
    def result():
        if request.method == 'POST':
            result = request.form
            Name = result['Name']
            name = Name
            PassengerId = int(result['PassengerId'])
            Pclass = int(result['Pclass'])
            Sex = result['Sex']
            Age = float(result['Age'])
            SibSp = int(result['SibSp'])
            Parch = int(result['Parch'])
            Ticket = int(result['Ticket'])
            Fare = float(result['Fare'])
            Cabin = result['Cabin']
            Embarked = result['Embarked']
            input_list = [PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]
            #pred, f_name, total_count = my_main(input_list)
            #
            # make a prediction
            #
            # new_x_data = get_new_data()
            xdata = [ [ PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked ] ]
            new_x_df = pd.DataFrame(data=xdata, columns = ['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
            new_x_data = new_x_df
            print("\n NEW_X_DATA\n", new_x_data, sep="")

            cleaned_new_x_data = prep_new_data(new_x_data)
            print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
            print("cleaned_new_x_data sample count", cleaned_new_x_data.shape[0])

            if cleaned_new_x_data.shape[0] > 0:
                # call the trained model to get a prediction
                # pass the new x sample and get a prediction back
                the_prediction = make_prediction(cleaned_new_x_data)

                print("\n original new_x_data =\n", new_x_data, sep="")
                print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
                print("\nThe predicted result = ", the_prediction)
            else:
                print("\n original new_x_data =\n", new_x_data, sep="")
                print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
                print("\nThe predicted result = NO PREDICTION due to no data samples")
			#
            # end of make a prediction
            #
            if the_prediction[0] == 1:
                prediction = "TRUE"
            elif the_prediction[0] == 0:
                prediction = "FALSE"
            else:
                prediction = "UNKNOWN - no or bad data sample"
                
		
            return render_template('template.html',
									my_string="titanic.csv",
										name=name,
										input=input_list,
											prediction=prediction)
	# end of function    
#     #
#     # make a prediction
#     #
#     new_x_data = get_new_data()
#     print("\n NEW_X_DATA\n", new_x_data, sep="")
    
#     cleaned_new_x_data = prep_new_data(new_x_data)
#     print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
#     print("cleaned_new_x_data sample count", cleaned_new_x_data.shape[0])
    
#     if cleaned_new_x_data.shape[0] > 0:
#         # call the trained model to get a prediction
#         # pass the new x sample and get a prediction back
#         the_prediction = make_prediction(cleaned_new_x_data)

#         print("\n original new_x_data =\n", new_x_data, sep="")
#         print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
#         print("\nThe predicted result = ", the_prediction)
#     else:
#         print("\n original new_x_data =\n", new_x_data, sep="")
#         print("\n CLEANED_NEW_X_DATA\n", cleaned_new_x_data, sep="")
#         print("\nThe predicted result = NO PREDICTION due to no data samples")
    


# In[ ]:


# ###############################################################################
# #
# # This is REALLY the main body of my program
# #
# ###############################################################################
# if __name__ == "__main__":
# 	print("++-->> Name mangled variable ... __name__::", __name__)
# 	print("++-->> ANALYZE_TITANIC_Logistic_Regression_APP.py program is being run STANDALONE!!\n\n")
	
# 	# instantiate the Flask object and then run it
# 	app = Flask(__name__)
	
# 	@app.route('/',methods = ['POST', 'GET'])
# 	def index():
# 		return render_template('input_template.html')
# 	# end of function
	
	
# 	@app.route('/result',methods = ['POST', 'GET'])
# 	def result():
# 		if request.method == 'POST':
# 			result = request.form
# 			name = result['name']
# 			Pclass = int(result['Pclass'])
# 			Sex = int(result['Sex'])
# 			Age = float(result['Age'])
# 			SibSp = int(result['SibSp'])
# 			Parch = int(result['Parch'])
# 			Fare = float(result['Fare'])
# 			Embarked = int(result['Embarked'])
# 			input_list = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
# 			pred, f_name, total_count = my_main(input_list)
			
# 			if pred == 1:
# 				prediction = "TRUE"
# 			else:
# 				prediction = "FALSE"
		
# 			return render_template('template.html',
# 									my_string=f_name,
# 										name=name,
# 										input=input_list,
# 											prediction=prediction,
# 												total_count=total_count)
# 	# end of function
	
# 	app.run(host='0.0.0.0', debug=True)
# else:
# 	print("++-->> Name mangled variable ... __name__::", __name__)
# 	print("++-->> ANALYZE_TITANIC_Logistic_Regression_APP.py program is being called by SOMEONE!!\n\n")
# # end of if else

# ###############################################################################
# # END OF FILE

