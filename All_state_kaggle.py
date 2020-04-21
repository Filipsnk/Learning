#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pandasql as sql

data = pd.read_csv('train.csv')

query = 'select * from data where Customer_ID in (10015208,10015208,10015221,10015226,10015288)'




customer_data = sql.sqldf(query, locals())


for i in data.columns:
    print(i,round(data[i].isnull().sum()/len(data[i]),2))


## Dla kazdego klienta wyciagnij jego finalna polise ktora kupil (ostatni wiersz per klient)
def klient(df):
        
    klient_zakup= pd.DataFrame()
    
    Customers=list(set(df['customer_ID']))
    
    for i in Customers:
        
        cust=df.loc[df['customer_ID']==i,:][-1:]
        
        klient_zakup = klient_zakup.append(cust)
        print('Dodawanie klienta numer: ',i)


    



