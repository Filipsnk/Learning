
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