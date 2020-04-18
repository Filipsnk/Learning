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

#JAROSLAW KROLEM POLSKI NA ZAWSZE#






