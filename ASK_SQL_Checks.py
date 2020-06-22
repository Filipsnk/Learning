
### SQL CHECKS ###

### CHECK OF RANDOM CUSTOMERS ### 

query = 'select * from train where Customer_ID in (10015208,10015221,10015226,10015288)'
customer_data = sql.sqldf(query, locals())

### CHECK IF THE NUMBER OF STATES IS EQUAL BETWEEN TRAIN AND TEST ###

query_1 = 'SELECT DISTINCT state FROM train'
query_2 = 'SELECT DISTINCT state FROM test'

distinct_states_train = sql.sqldf(query_1, locals())
distinct_states_test = sql.sqldf(query_2, locals())

check_1 = 'SELECT state FROM train WHERE state NOT IN (SELECT state FROM test)'
check_1_result = sql.sqldf(check_1, locals())

### CHECK HOW MANY NULLS ARE THERE IN RISK_FACTOR ###

risk_nulls = 'SELECT COUNT(*) FROM train WHERE risk_factor is NULL'
risk_nulls_result = sql.sqldf(risk_nulls, locals())

### CHECK DEPENDENCY BETWEEN RISK_FACTOR AND AGE_OLDEST / YOUNGEST ###

risk_factor_nulls = 'SELECT risk_factor, MAX(age_oldest), MIN(age_oldest), MAX(age_youngest), MIN(age_youngest) FROM train'
risk_factor_nulls_2 = 'SELECT risk_factor, AVG(age_oldest), AVG(age_youngest) FROM train GROUP BY risk_factor'
risk_factor_nulls_results = sql.sqldf(risk_factor_nulls, locals())
risk_factor_nulls_2_results = sql.sqldf(risk_factor_nulls_2, locals())

### CHECK A DEPENDENCY ON CAR_AGE ###

option_A = 'SELECT A, AVG(car_age) FROM train GROUP BY A'
option_A_results = sql.sqldf(option_A, locals())

option_A_age = 'SELECT A, AVG(age_oldest), AVG(age_youngest) FROM train GROUP BY A'
option_A_age_results = sql.sqldf(option_A_age, locals())

### CHECK MINIMUM AND MAXIMUM AGE ###

hour_check = 'SELECT MAX(hour), MIN(hour) FROM train'
print(sql.sqldf(hour_check, locals()))

### CAR_VALUE VARIABLE VALUES ###
car_value_check = 'SELECT DISTINCT car_value FROM train'
print(sql.sqldf(car_value_check, locals()))

### CODE FOR GROUPPING STATES (NORTH/SOUTH/EAST/WEST) ###

state_check = 'SELECT DISTINCT state FROM train'
print(sql.sqldf(state_check, locals()))

### CORRELATIONS BETWEEN INSURANCE OFFERS FOR DEFINING SOME RULES, FOR EXAMPLE
### IF A == 1, THEN B == 2 ETC.

columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
insurance_corr = train[columns].corr()
sns.heatmap(insurance_corr)

# Most important pairs are: C-D (0,64); A-F (0,53) and B-E (0,46)

pd.crosstab(train['C'],train['D'],margins = False) # if C == 3 -> D == 3
pd.crosstab(train['A'],train['F'],margins = False) # if A == 0 -> F == 0
pd.crosstab(train['B'],train['E'],margins = False) # if B == 0 -> E == 0
                                                   # if B == 1 -> E == 1
# Some random attempts
pd.crosstab(train['A'],train['E'],margins = False) # if A == 0 -> E == 0
ref_column = 'G'
for col in columns:
    print(col + ' vs. column {}'.format(ref_column))
    print(pd.crosstab(train[col],train[ref_column],margins = False).apply(lambda r: r/r.sum(), axis=1).max().max())