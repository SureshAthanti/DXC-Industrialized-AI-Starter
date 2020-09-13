import pandas as pd
import janitor #data cleaning
from ftfy import fix_text #data cleaning
import nltk #data cleaning
nltk.download('punkt') #data cleaning
import scrubadub #data cleaning
import arrow #normalizing dates
import numpy as np
from sklearn.base import TransformerMixin
from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder
from dxc.ai.global_variables import globals_file


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def encode(data):

    '''function to encode non-null data and replace it in the original data'''
    encoder = OrdinalEncoder()
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #encoders_store[column_name]=encoder
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return (data,encoder)

def impute(df):
    # imputer = KNN()
    imputer = KNN(k=2)
    object_types = list(X.select_dtypes(include=['object']).columns)
    num_types = list(set(X.columns) - set(object_types))
    encoders_store={}
    for column in num_types:
        skew=df[column].skew()
        if (-1 < skew < 1):
            df[column]=df[column].fillna(df[column].mean())
        else :
            df[column]=df[column].fillna(df[column].median())
    #create a for loop to iterate through each column in the data
    for columns in object_types:
        new=encode(df[columns],columns)
        encoders_store[columns]=new[1]
    imputed_data = pd.DataFrame(np.round(imputer.fit_transform(df)),columns = df.columns)
    for columns in object_types:
        imputed_data[columns]=encoders_store[columns].inverse_transform(np.array(imputed_data[columns]).reshape(-1,1))
    return imputed_data
    
#CLEANING FILE
def clean_dataframe(df, impute = False, text_fields = [], date_fields = [], numeric_fields = [], categorical_fields = []):   
    clean_df = (
      df
      #make the column names lower case and remove spaces
      .clean_names()

      #remove empty columns
      .remove_empty()

      #remove empty rows and columns
      .dropna(how='all')
    )

    #remove harmful characters. remove personal identifiers. make lowercase
    for field in text_fields:
        field = '_'.join(field.split()).lower()
        clean_df[field] = clean_df[field].fillna(' ').apply(fix_text)
        clean_df[field] = clean_df[field].apply(scrubadub.clean, replace_with='identifier')
        clean_df[field] = clean_df[field].str.lower()
  
    #impute missing values
    if impute:
        clean_df = DataFrameImputer().fit_transform(clean_df)

    #standardize the format of all date fields
    for field in date_fields:
        field = '_'.join(field.split()).lower()
        clean_df[field] = clean_df[field].apply(arrow.get)

    #make sure all numeric fields have the proper data type
    for field in numeric_fields:
        field = '_'.join(field.split()).lower()
        clean_df[field] = pd.to_numeric(clean_df[field])
  
    #make sure all categorical variables have the proper data type
    for field in categorical_fields:
        field = '_'.join(field.split()).lower()
        clean_df[field] = clean_df[field].astype('category')
    
    clean_df=clean_df.clean_names()
    
    globals_file.clean_data_used = True

    return(clean_df)

