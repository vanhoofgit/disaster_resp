#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:47:05 2020

@author: benedictusvanhoof
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    input : two csv files
    returns a merged dataframe of the two csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on = 'id', how = 'inner')
    return df

## this function creates a list of columns which have only zero values in it
def look_zero_columns (dataframe):
    '''
    input : a dataframe
    returns a list of the column names that have only zero values
    '''
    only_zero_columns = []
    dataframe = dataframe.iloc[:,4:]
    df_sum = pd.DataFrame(dataframe.sum())
    df_sum = df_sum.reset_index()
    df_sum.columns = ['category','number']
    for i in range(df_sum.shape[0]):
        if df_sum.iloc[i,1]==0:
            only_zero_columns.append(df_sum.iloc[i,0])                                                      
    return only_zero_columns

## This function deletes the columns with only  zero values
def drop_zero_columns (dataframe,column_list):
    '''
    input : a dataframe and a list with the column names with only zero values
    returns the dataframe with these columns dropped
    '''
    for name in column_list:
        dataframe.drop(columns = [name], axis = 1, inplace=True)
    return dataframe


def clean_data(df):
    
    '''
    Input : A non-cleaned dataframe
    Output: The  same dataframe but cleaned
    
    '''
    ## change od the dtype of the id column
    df['id']= df['id'].astype(str)
    ## The categories column is expanded with the values between the semicolons
    categories_temp = df.categories.str.split(";",expand=True) 
    ## The values of the first row are taken as base for the headers of the categories_temp        
    ## dataframe
    header_list= categories_temp.iloc[0,:]
    column_list = []
    ## The text before the - sign is chosen as column header
    for header in header_list:
        index = header.find("-")
        column_header = header[0:index]
        column_list.append(column_header)
    categories_temp.columns = column_list
    ## The cells in the categories_temp dataframe are replaced by the integer conversion of the    
    ## last value of each cell
    for column in categories_temp.columns:        
        categories_temp[column] = categories_temp[column].apply(lambda x : int(x[-1]))
    ##replace the values 2 in the related column with the mode of that column
    ## calculated the mode
    my_modus = categories_temp['related'].mode()
    categories_temp.loc[:,'related'].replace([2],my_modus,inplace =True)
    
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    ## Concatenate the df dataframe and the categories_temp dataframes
    df = pd.concat ([df,categories_temp], axis = 1 , join = 'inner')
    ## print out the number of duplicates
    number_rows = df.shape[0]
    print ("number of rows in the dataframe : ", number_rows)
    number_duplicates = df.duplicated().sum()    
    print ("number of duplicates in the dataset : ", number_duplicates)
    
    ## remove the duplicates
    df.drop_duplicates(inplace = True)
    number_rows = df.shape[0]
    print ("number of rows in the dataframe after deleting duplicates : ", number_rows)
    
    ## Delete the rows with a zero value for the 'related' field
    ## They have in all the columns a zero value and are not related to any disaster
    #df = df[df['related'] == 1]
    #number_rows = df.shape[0]
    #number_columns = df.shape[1]
    #print ("number of rows in the dataframe after deleting the lines with a zero value for related : ", number_rows)
    
    ## Drop the column 'related'. It has on every row the value 1 and will create a bias towards that column
    #print ("number of columns in the dataframe before deleting the column 'related' : ", number_columns)
    #df.drop(columns = ['related'], axis =1, inplace =True)
    #number_columns = df.shape[1]
    #print ("number of columns in the dataframe after deleting the column 'related' : ", number_columns)
    
    ## Dop the column(s) with only zero values
    #1) call of the funcion look for the column(s) with only zero
    #my_zero_columns = look_zero_columns (df)
    #print ("Columns with only zeros found : ", my_zero_columns)
    #2) call of the function to drop the columns
    #drop_zero_columns(df,my_zero_columns)
    #number_columns = df.shape[1]
    #print ("number of columns in the dataframe after deleting the columns with only zeros : ", number_columns)
    
    
    
    
    
    return df
                

def save_data(df, database_filepath):
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    df.to_sql('DisasterMessages', engine,if_exists='replace' ,index=False)
    
    

def main():   
     
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

              
        my_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        
        my_df = clean_data(my_df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        
        
        save_data(my_df, database_filepath)
        
        print('Cleaned data saved to database!')
       

    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()   