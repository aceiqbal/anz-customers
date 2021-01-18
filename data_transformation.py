#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="map")


#Reading the Dataset
dataset = pd.read_csv('anz.csv')

#Cleaning the Dataset
dataset_rev = dataset.drop(['status', 'bpay_biller_code', 'account', 'currency', 'merchant_id', 'merchant_code', 
                            'first_name', 'merchant_suburb', 'extraction', 'transaction_id', 'country', 
                            'merchant_long_lat', 'merchant_state'], axis=1)



#Transforming Customer Dataset
from datetime import datetime

cus_dataset = pd.DataFrame(columns = ['customer_id', 'monthly_expenses', 'balance', 'age', 'gender', 'state', 'card_freq', 'salary'])

for customer in dataset_rev['customer_id'].unique():
    df = dataset[dataset['customer_id'].str.contains(customer)]
    salary = 0
    
    #Handling Nan= Values
    df['card_present_flag'] = dataset_rev['card_present_flag'].replace(np.nan, 0)
    
    #Find the customers Current Balance
    rec_date = datetime.strptime('01/01/1800', '%m/%d/%Y')
    for i in df.index.values:
        lst_bal = datetime.strptime(df['date'][i], '%m/%d/%Y')
        if lst_bal > rec_date:
            balance = df['balance'][i]
            
    #Changing the units of time into month blocks
    for i in df.index.values:
        time = datetime.strptime(df['date'][i], '%m/%d/%Y')
        df['date'][i] = str(time.month) + '-' + str(time.year)
    
    #Determining customer salary after Taxes
    df_salary = df[df['txn_description'].str.contains('PAY/SALARY')]  
    if len(df_salary) > 0:
        df_salary = df_salary[df_salary['date'].str.contains(df_salary['date'].unique()[1])]
        if len(df_salary) > 3:
            salary = df_salary['amount'].values[0] * 52
        elif 3 > len(df_salary)  > 1:
            salary = df_salary['amount'].values[0] * 26
        elif len(df_salary) == 1:
            salary = df_salary['amount'].values[0] * 12

    #Determining monthly expenses
    df_expenses = df[df['movement'].str.contains('debit')]  
    monthly_expenses = []
    for i in df['date'].unique():
        df_month_expense = df_expenses[df_expenses['date'].str.contains(i)]
        monthly_expenses.append(df_month_expense['amount'].sum())
    monthly_expenses = np.average(monthly_expenses)
    
    #Determine Percentage of Card Purchases
    card_freq = df_expenses['card_present_flag'].sum() / len(df_expenses)
    
    #Find Customer State
    lon_lat = df['long_lat'].values[0].split(' ')
    try:
        location = geolocator.reverse(lon_lat[1]+' '+lon_lat[0])
        state = location.raw['address']['state']
    except:
        state = None
    
    cus_dataset = cus_dataset.append({'customer_id': customer, 'monthly_expenses': monthly_expenses, 
                         'balance': balance, 'age': df['age'].values[0], 'gender': df['gender'].values[0], 
                         'state': state, 'card_freq': card_freq, 'salary': salary}, ignore_index=True)

#Clean New Dataset
cus_dataset = cus_dataset.dropna()
    


    
