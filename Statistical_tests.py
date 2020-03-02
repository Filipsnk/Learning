#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICAL TESTS

1. NORMALITY TESTS
2. CORRELATION TESTS
3. STATIONARY TESTS
4. PARMAETRIC STATISTICAL HYPOTHESIS TESTS
5. NON PARAMETRIC STATISTICAL TESTS




Normality tests
Shapiro- Wilk test

Assumptions
    1. observations are independent

Interpretation:
    H0: the sample has a Gaussian Distribution
    H1: the sample does not have a Gaussian Distribution
"""

from scipy.stats import shapiro
import matplotlib.pyplot as plt
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
plt.hist(data)

stat,p = shapiro(data)

print('stat= %.3f, p= %.3f' %(stat,p))

if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


"""
D'Agostino K^2 test

Assumptions:
    1. Observations in each sample are independent

Interpretation:
    
    H0: the sample has a Gaussian Distribution
    H1: the sample does not have a Gaussian Distribution
"""

from scipy.stats import normaltest

stat, p = normaltest(data)

print('stat =%.4f, p=%.4f' %(stat,p))

if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')

"""
Anderson Darling Test

Assumptions:
    1.Observations in each sample are independent

Interpretation:
    
    H0: the sample has a Gaussian Distribution
    H1: the sample does not have a Gaussian Distribution
"""

from scipy.stats import anderson

result = anderson(data)

print('Anderson stat= %.4f' %(result.statistic))

result.critical_values
result.significance_level

for i in range(len(result.critical_values)):
    
        a,b = result.critical_values[i],result.significance_level[i]
        
        if result.statistic < a:
            print('Probably Gaussian distribution at the %.1f%% level' %(b))
        else:
            print('Probably not a Gasussian distribution at the %.1f%% level' %(b))
            
"""
Correlation test

Pearson correlation test

Assumptions:
    1. Observations in each sample are independent and identically distributed
    2. Observations in each sample are normally distributed
    3. Observations in each sample has the same variance
    
Interpretation:
    
    H0: the two samples are idependent
    H1: there is a dependency between the samples
    
"""

from scipy.stats import pearsonr

data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]

plt.hist(data1)
plt.hist(data2)

stat, p = pearsonr(data1,data2)

print('stat= %.3f, p-value= %.3f' %(stat,p))

if p > 0.05:
    print('Probably independent')
else:
    print('Not idependent')




## generate data until normally distributed
    
random_data = []
normal = False
count = 0
data = []
k = 0


while k < 10:
    
    for i in range(0,10):
        data = random.random()
        random_data.append(data)
    
    stat, p = shapiro(random_data)
        
    if p > 0.05:
        print('Data is normally distributed. Well done')   
        break
    else:
        print('failed')
        random_data = []
    k+=1
    
    
"""
Spearman's Rank Correlation

Assumptions
    1. Observations in each sample are independent and identically distributed

Interpretation:

    H0: two samples are independent
    H1: there is a dependency between the samples
"""
from scipy.stats import spearmanr    

data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]

stat, p = spearmanr(data1, data2)

print('stat= %.4f, p-value= %.4f' %(stat,p))

if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')

"""
Kendal Rank Correlation

Test whether two samples have a monotonic relationship

Assumptions:
    1) observations in each sample are independent and identically distributed
    2) observations in each sample can be ranked
    
Interpretation:
    1) two samples are independent
    2) there is a dependency between the samples
"""
from scipy.stats import kendalltau

data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]

stat, p = kendalltau(data1, data2)

print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
    


"""
Chi squared test

tests whether two categorical variables are related or independent

Assumptions:
    1) obsevations used in the calculation of the contingency table are independent
    2) 25 or more examples in each cell of the contingency table

Interpretation:
    H0: the two samples are independent
    H1: there is dependency between the samples
"""
   
from scipy.stats import chi2_contingency

table = [[10, 20, 30],[6,  9,  17]]

stat, p, dof, expected = chi2_contingency(table)

print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')    

"""
Parametric statistical tests

Student's t-test

This tests check whether the means of two independent samples are significantly different

H0: the means of two samples are equal
H1: the means of two sample are not equal
"""


from scipy.stats import ttest_ind
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


"""
ANOVA

tests whether the means of two or more independent samples are significantly different

Assumptions:
    1) Observations in each sample are independent and identically distributed
    2) observations in each sample are normally distributed
    3) observations in each sample have the same variance
    
Interpretation:
    
    H0: the means of the sample are equal
    H1: one or more of the means of the samples are unequal
"""

from scipy.stats import f_oneway
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
    
"""

Non parametric test

Mann-Whitney U test

Tests whether the distributions of two samples are equal

Assumptions:
    1) Observations are independent
    2) Observations in each sample can be ranked


Interpretation:

    H0: the distributions of both samples are equal
    H1: the distributions of both samples are not equal

"""
from scipy.stats import mannwhitneyu
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
    
    
"""

Wilcoxon Signed-Rank Test

Tests whether the distributions of two paired samples are equal or not.

Assumptions

Observations in each sample are independent and identically distributed (iid).
Observations in each sample can be ranked.
Observations across each sample are paired.
Interpretation

H0: the distributions of both samples are equal.
H1: the distributions of both samples are not equal.

"""
 from scipy.stats import wilcoxon
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = wilcoxon(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
    
"""
Kruskal-Wallis H Test

Tests whether the distributions of two or more independent samples are equal or not.

Assumptions

Observations in each sample are independent and identically distributed (iid).
Observations in each sample can be ranked.
Interpretation

H0: the distributions of all samples are equal.
H1: the distributions of one or more samples are not equal.
"""
from scipy.stats import kruskal
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
    
"""
Friedman Test

Tests whether the distributions of two or more paired samples are equal or not.

Assumptions

Observations in each sample are independent and identically distributed (iid).
Observations in each sample can be ranked.
Observations across each sample are paired.
Interpretation

H0: the distributions of all samples are equal.
H1: the distributions of one or more samples are not equal.

"""
from scipy.stats import friedmanchisquare
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')








