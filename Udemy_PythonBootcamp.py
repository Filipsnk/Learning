#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 08:30:33 2019

@author: filipfraczek
The complete Python Programmer Bootcamp

"""




liczba1 = int(7)
liczba2 = str(7)

print(liczba2)

word = 'summer'

guess = input(str('I\'m thinking about the season. Can you guess what is this? \
it\'s a season.>>> '))

if guess =='summer':
    print('well done')
elif guess =='winter':
    print('idiot')
elif guess =='autumn':
    print('kretyn')
else:
    print('idiota')
        
num = str(input('Wprowadz liczbe miedzy 1 a 5 slownie: '))
num_o = num.lower()


if num == 1:
    print('jeden')
elif num == 2:
    print('dwa')
elif num == 3:
    print('trzy')
elif num == 4:
    print('cztery')
elif num < 1 or num > 5:
    print('zly zakres')
else:
    print('piec')

if guess.isdigit():
    print('yes')


name = str(input('Wprowadz swoje imie: '))

if len(name) > 5:
    print('dobry wynik. Twoja dlugosc imienia to',len(name))
else:
    print('it\' a secret')


### for loops###

# ostatni argument mowi co ile ma byc print
for i in range(0,10,2):
    print(i)

data = [1,2,3,4,5]

total = 0

for num in data:
    total = total + num
print(total)

sum(data)


## find max

find_max = 0

for i in data:
    if i > find_max:
        find_max = i
print(find_max)

#Function sort

data_copy = [53,76,25,98,56,42,69,81]

for i in range(0,len(data_copy)):
    for j in range(0,len(data_copy)-i-1):
        if data_copy[j] > data_copy[j+1]:
            data_copy[j],data_copy[j+1] = data_copy[j+1],data_copy[]
      
        
#Sorted function without using a function

l = [64, 25, 12, 22, 11, 1,2,44,3,122, 23, 34]

for i in range(len(l)):
    for j in range(i+1,len(l)):
        if l[i] > l[j]:
            l[i],l[j] = l[j],l[i]
print(l)
sorted(l)

#append a list

my_list = [1,2]

my_list.append(1)
my_list.remove(1)

## to add new list into list
my_list.extend([1,2,3,4])

## While loops

user_input = int(input('wprowadz wiek: '))
ages = []

while user_input > 0:
    ages.append(user_input)
    user_input = int(input('powiedz jaki wiek: '))
print(ages)

## how many people in class

count = 0
name = input('wprowadz imie, jesli chcesz skonczyc wpisz n: ')
klasa = []

while name != 'n':
    count += 1
    klasa.append(name)
    print(f'zostalo dodane imie {name}')
    name = input('wprowadz kolejne imie')
    
print(f'W klasie jest {count} uczniow. Natomiast imiona to {klasa}')

### sprawdz ktore liczby sa podzielne

liczba = [10,20,30,41,22,6]

for i in liczba:
    if i % 2 == 0:
        print(i)

# exercises
"""
ask the user for two numbers between 1 and 100. then count from the
lower number to the higher number. Print the results to the screen
"""

lower = int(input('wprowadz liczbe: '))
upper = int(input('wprowadz liczbe: '))

for i in range(lower, upper + 1):
    print(i, end = ' ')

'''
ask the user to inputa a string and then print it out to the screen in reverse order
'''

word = str(input('wprowadz slowo: '))

for i in word[::-1]:
    print(i, end= ' ')

'''
ask the user for a number between 1 and 12 and then display a times table for that number
'''

num = input('wprowadz liczbe: ')

while (not num.isdigit()) or (int(num) > 12) or (int(num)<1):
    print('musi byc int pomiedzy 1 oraz 12')
    num = input('wprowad liczbe')
num = int(num)
print('==============================')
print()
for i in range(1,13):
    print(f'{i} x {num} = {i *num}')
    
'''
Ask the user to input a sequence of numbers. Then calculate the mean and print results
'''

numbers = []

liczba = input('wprowadz liczbe, zeby skonczyc wpisz exit: ')

while liczba != 'exit':
    while not liczba.isdigit():
        print('wprowadz inta')
        liczba = input('wprowadz liczbe: ')
    numbers.append(int(liczba))
    liczba = input('wprowadz kolejna liczbe: ')
ile = 0

for i in numbers:
    ile = ile + i
    
print(f'Srednia z Twoich liczb wynosi {ile/len(numbers)}')


'''
calculate factorial - 15... in other words 5x4x3x2x1
'''

licznik = 1
for i in range(1,16):
    licznik = licznik * i
print(licznik)


'''
Calculate fibonacci series . first 20

series -> 0,1,1,2,3,5,8....
'''

suma = []
a = 0
b = 1

for i in range(10):
    suma.append(a)
    a,b = b, a+b
print(suma)


##
1. 0
a = 0, b = 1
2.
a = 1 b = 1
3.
a = 1 b = 2
4
a = 2 b = 3
5
a = 3 b = 5





def Fibonacci(n): 
    if n<0: 
        print("Incorrect input") 
    # First Fibonacci number is 0 
    elif n==1: 
        return 0
    # Second Fibonacci number is 1 
    elif n==2: 
        return 1
    else: 
        return Fibonacci(n-1)+Fibonacci(n-2) 

###################################### Modules
        
import math

#to check what functions are inside module - math
dir(math)

## webbrowser

import webbrowser

webbrowser.open(....)

##################################### Dictionaries

lista = {'Polska':'Warszawa','Niemcy':'Berlin'}

lista['Polska']
lista.get('Polska',0)

## to add an item into dictionary

lista['Wlochy'] = 'Rzym'

lista.keys()
lista.values()
lista.items() ## zwraca tuple
lista

for country, city in lista.items():
    print(f' kraj:{country} oraz stolica{city}')

print(lista.get('Portugalia',0)+1)

tekst = '''Siemaneczko iiiiiii '''

lista = {}

for i in tekst:
    print(i)
    lista[i.lower()] = lista.get(i,0)+1



#### Manual sorting in descending order
    
l = [60,2,13,10,1,5,20]

for i in range(0,len(l)):
    for j in range(i+1,len(l)):
        if l[i] > l[j]:
            l[i],l[j] = l[j],l[i]
print(l)

#### Modules ### Importowanie bibliotek
### How many numbers in text ###


import numpy as np
import math
import pandas as pd
# to check what inside math module
dir(math)

slownik = '''
Przyszedl taki dzien
"Filip ma jeden"
Kuba ma dwa
'''

pusta = pd.DataFrame(columns=['Litera','Liczba'])

slownik.count('z')

pusta_2 = {}

for letter in slownik:
    pusta_2[letter] = slownik.count(letter)
    

for letter in slownik:
    if letter not in pusta['Litera'].values:
        pusta = pusta.append({'Litera': letter, 'Liczba': 1}, ignore_index = True)
    else:
        pusta.loc[pusta[pusta['Litera'] == letter].index,'Liczba']+=1
    
    
########## Zip function
        
a = [1,2,3,4]
b = ['a','b','c','d']
        
nowa = list(zip(a,b))    


### to unpack a zip
i,j = zip(*nowa)

## Difference between tuples and list
my_list = [1,2,3,4]

my_list[0] = 1000
print(my_list)

#Tuple
tuple_1 = (1,2,3)
tuple_1[0] = 1000 ----- '''can not add any other element '''


### 
countries = {'Polska':{'Stolica':'Warszawa','jezyk':'Polski'},'Wlochy':{'Stolica':'Rzym','jezyk':'Wloski'}}

countries.items()

countries.keys()

for kraj,items in countries.items():
    print(f'{items["Stolica"]} is a capital of {kraj}. Tam mowia po {items["jezyk"]}')


## Dictionary & List Comprehension
    
M = [x**2 for x in range(1,11)]
print(M)

L = []
for i in range(1,11):
    L.append(i**2)

#### Exercise
    
test = {'Polska':'Warszawa','Wlochy':'Rzym'}

podac = str(input('Podaj kraj: '))
    
if test.keys() == podac:
    print('brawo')
else:
    print('slabo')


## Write a code that represent first 12 Fibonacci values (as dictionary)
    
n = 12
a = 0
b = 1
fib = dict()

for i in range(n):
    fib[i] = a
    a,b = b, a+b
print(fib) 


## Fibonacci as a df

n = 10
a = 0
b = 1
test_2 = []

for i in range(n):
    test_2.append(a)
    a,b = b, a+b
print(test_2)   

## 
import datetime
import random
import matplotlib.pyplot as plt
import random

today = datetime.date.today()
holiday  = datetime.date(2019,12,24)
remain = today - holiday

print(f'Do wakacji zostalo{remain}')

### exercie

tekst = 'abdkajsdkjahdkahdkjsadhasd'
numer = random.randint(1,len(tekst))

scal = dict(zip(numer,tekst))
tabela = dict()

for i in tekst:
    tabela[i] = random.randint(1,100)
    
x,y = zip(*tabela.items())
plt.bar(x,y)

#################FILES###############

f = open('test.txt','w')

###
w - write
r - read
a - append
###

print(type(f))

f.write('siema ziomeczku co tam slychac,\n')

f.close

f = open('test.txt','r')
print(f.read())
print(f.readline())

# append

f = open('test.txt','a')
f.write('Siema, co tam slychac')
f.close

f = open('test.txt','r')
f.read()

### another way. We do not close a file each time. very useful

with open('test.txt','r') as f:
    for line in f.readlines():
        print(line)

## Functions

def powitanie(imie,nazwisko):
    print('Siema ',imie,'. Na drugie imie masz ',nazwisko)

## Create a fibonacci function
    
n = 10
a = 0
b = 1

for i in range(n):
    print(a)
    a,b = b,a+b
    
def fib(liczba):    
    a = 0
    b = 1
    for i in range(liczba):
        a,b = b, a+b
        print(a)
        
## if we do not know how many elements will be in 'remainder' ###
def calc_mean (first, *remainder):
    mean = (first + sum(remainder))/(1+ len(remainder))
    return mean

#####  Classes
    
y = 1
dir(y)


class Medical(object):
    
    status = 'Patient'
    
    def __init__(self,name,age):
        self.name = name
        self.age = age
        self.conditions = []
        
    def info(self):
        print(f'The name of this patient is {self.name}, however his age is {self.age}'\
              f'Additional information {self.conditions}')
        
    def add_info(self,information):
        self.conditions.append(information)
        
class infant(Medical):
    
    def __init__(self,name,age):
        self.vaccination = []
        super().__init__(name,age)
        
    def add_vac(self,vacine):
        self.vaccination.append(vacine)
    
    def get_info(self):
        print(f'Patient record: {self.name}, {self.age} years' \
              f'Patient has had {self.vaccination} vaccines' \
              f'Current information: {self.conditions}')



### exercise
    
class BankAccount(object):
    
    def __init__(self,balance = 0.0):
        self.balance = balance
    
    def status(self):
        print(f'Twoj obecny stan konta wynosi {self.balance}')
    
    def doladuj(self):
        kwota = int(input('Wprowadz kwote ktora doladujesz konto: '))
        self.balance += kwota
        print(f'Twoj status konta wynosi obecnie:{self.balance}')
        
    def wyciagnij(self):
        kwota_2 = int(input('Wprowadz kwote ktora chcesz wyciagnac: '))
        
        if kwota_2 <= self.balance:
            self.balance -= kwota_2
            print(f'Twoj rachunek biezacy wynosi: {self.balance}')
        else:
            print(f'Nie masz wystarczajacych srodkow na koncie')
        
        
# create a circle class that will take the value of radius and return the area of the circle
            
from math import pi

class circle(object):
    
    def __init__(self,radius):
        self.radius = radius
        
    def pole(self):
        kolo = (self.radius * pi)
        print(f'Pole kola wynosi {kolo}')



##Cesar Cipher exercise
        
alfabet = 'abcdefghijklmnopqrstuvwxyz'
slowo = 'siema'       


def funkcja(tekst,liczba):
    
    liczba = int(liczba)
    slowo = tekst
    
    if liczba < len(alfabet):
        
        for char in slowo:
            indeks = alfabet.find(char)
            wynik = alfabet[indeks + liczba]
            print(wynik)
    
    else:
        
        for char in slowo:
            indeks = alfabet.find(char)
            wynik = alfabet[(indeks + liczba)%len(alfabet)]
            print(wynik)
        
        
def funkcja_2(tekst,liczba):
    
    liczba = int(liczba)
    slowo = tekst
    
    for char in slowo:
        indeks = alfabet.find(char)
        wynik = alfabet[(indeks + liczba)%len(alfabet)]
        print(wynik)
        
            
 ### Two sum example

2,3,4,5

sum -> 10

2,3
2,4
2,5
3,4
3,5
4,5

nums = [2,8,4,3,2,1,1,1]


def suma(nums,target):

    d ={}

    for i in range(len(a)):
        if target - nums[i] in d:
            print(d)
            return [d[target - nums[i]],i]
        else:
            d[nums[i]] = i

suma(nums,10)

##### matplotlib

import numpy as np
import matplotlib.pyplot as plt
from random import choice
%matplotlib inline  ### jesli chcemy aby wykresy byly pokazywane wewnatrz jupyter notebook###

dir(plt) 

plt.rc('figure',figsize = (12,6))
        
        
values_y = list(range(0,55,5))  
plt.plot(values)

values_x = np.arange(0,1.1,0.1)
        

plt.plot(values_x,values_y)
plt.xlim(0,1.0)
plt.ylim(0,max(values_y))
plt.title('Out plot')      
plt.xlabel('This is a horizontal plot')
plt.ylabel('Vertical axis')
plt.show()
        
#### create a random walk

step = [-1,1]  
step_choice = choice(step)      
  
random_walk = []

def kroki(ilosc_krokow):
    
    walk = []
    
    step_choice = choice([-1,1])
    
    walk.append(step_choice)
    
    for step in range(1,ilosc_krokow):
        
        kolejny = walk[step-1] + choice([-1,1])
        
        walk.append(kolejny)
    
    return walk
        
random_w = kroki(100)

print(len(random_w))

def plot_walk(walk):
    
    plt.plot(walk)
    plt.xlabel('Number of steps')
    plt.ylabel('Distance from origin')
    plt.title('Our random of walk')
    plt.show()
    
plot_walk(kroki(100))


def many_random(num,krokow):
    
    lab_list = list(range(1,num+1))
    
    x = list(range(1,krokow + 1))
    
    for i in range(0,num):
        
        plt.plot(kroki(krokow), label = 'Plot number ' + ' ' + str(lab_list[i]))
        plt.xlabel('Number of steps')
        plt.ylabel('Distance from Origin')
        plt.title('Out random of walk')
        plt.legend(loc = 'lower left')
    plt.show()
        
##### sierpinski triangle
    
''' p must be vector'''
    
def transf_1(p):
    
    x = p[0]
    y = p[1]
    
    x1 = 0.5 * x
    y1 = 0.5 * y
    
    return x1,y1

def transf_2(p):
    
    x = p[0]
    y = p[1]
    
    x1 = 0.5 * x + 0.5
    y1 = 0.5 * y + 0.5
    
    return x1,y1

def transf_3(p):
    
    x = p[0]
    y = p[1]
    
    x1 = 0.5 * x + 1
    y1 = 0.5 * y
    
    return x1,y1

transformations = [transf_1, transf_2, transf_3]

a1 =[0]
b1 =[0]

a,b = 0,0


for i in range(10000000):
    
    trans = choice(transformations)
    a,b = trans((a,b))
    
    a1.append(a)
    b1.append(b)
    
plt.rc('figure',figsize = (16,16))

plt.plot(a1,b1,'o')


a = [1,2,3]

a.pop()

###################

lista = 'abcdef'
lista_2 = list(lista)

for i in lista_2:
    print(i)
    
for i in range(len(lista_2)):
    print(i, lista_2[i])
    
## ENUMERATE

for i,probka in enumerate(lista_2):
    print(i,probka)
    
    
### sets
    
A = [1,1,2,2,3,3]
A = set(A)
A
# zwraca tuple - unikalne
 
B = [1,1,4,4,5,6,7]   
B = set(B) 
  
A & B #intersection
A | B #union
A - B #Differnece
A ^ B  #Symmetric difference

lista =[1,2,3,4,5,6]


def znajdz(param,lista):
    
    i = 0
    found = False
    
    while i < len(lista) and found == False:
        
        if lista[i] == param:
            found = True
        else:
            i = i + 1
    return found

test = [1,2,3,4,5,6]  
    
    
######### Insertion list

test = [1,2,3,4,5,6]

def znajdz(my_list):
    
    for i in range(1,len(my_list)):
        value = my_list[i]
        j=i
        while j > 0 and my_list[j-1] > value:
            
            my_list[j] = my_list[j-1]
            
            j = j-1
            
        my_list[j] = value
    return my_list
        

#### Linear search
    

''' if 6 in a list [1,5,2,4,1,2,3,4]'''

def linear(lista,wartosc):
    
    value = int(wartosc)
    found = False
    
    count = 0
    
    for i in range(0,len(lista)):
        
        if lista[i] == value:
            
            found = True
        else:
            
            count += 1
    return found,count
        
print(linear([6,5,8,2,3,45,87,24,70],87))
print(linear([6,5,8,2,3,45,87,24,70],88))
       
    
## z petla while

def linear_2 (lista,wartosc):
    
    value = int(wartosc)
    found = False
    i=0
    
    while i < len(lista) and found == False:
        
        if lista[i] == wartosc:
            found = True
        else:
            i += 1
    return found
    
## binary search

def binary (lista,wartosc):
    
    
    first = 0
    last  = len(lista)-1
    counter = 0
    found = False
    wartosc = int(wartosc)
    
    while first <= last and found == False:
        
        middle = int((first + last)/2)
        
        if lista[middle] == wartosc:   
            found = True
        else: 
            if lista[middle] > middle:
                last = middle - 1
            else:
                first = middle + 1 
    return found
        
        
######### credit card ## Luhns algorithm
    

lista = [3,7,1,4,4,9,6,3,5,3,9,8,4,3,1]


parzyste = []
nieparzyste = []
wynik = []
wynik_2 = []


def sprawdz_konto(lista):
    
    for i in range(1,len(lista)):
        
        if i % 2 == 0:
            
            new = lista[i]*2
            print(new)
            
            if new > 9:
                for j in str(new):
                    parzyste.append(int(j))
            else: parzyste.append(new)
            
        else:
            
            new_2 = lista[i]
            nieparzyste.append(new_2)
            
    wynik = sum(parzyste) + sum(nieparzyste)
    
    if wynik % 10 == 0:
        
        print('Card is valid')
    
    else:
        print('Card is not valid')
    
    
## podejscie dwa
        
parzyste_2 = []
nieparzyste_2 = []
parzyste =[]
wynik_2 = 0

def sprawdz_konto_2 (lista):
    
    parzyste = lista[1::2]
    
    for i in range(0,len(parzyste)):
        
        new_2 = parzyste[i]*2
        
        if new_2 > 9:
            for k in str(new_2):
                parzyste_2.append(int(k))
        else: parzyste_2.append(new_2)
    
    nieparzyste_2 = lista[::2]
     
    wynik_2 = sum(parzyste_2) + sum(nieparzyste_2)
     
    if wynik_2 % 10 ==0:
         
        print('Card is valid')
    else: print('Card is not valid')
     

### coins
    

n = 10

coins = [0] * n

for i in range(1,n):
    for j in range(0,n,i):
        coins[j] = 1 - coins[j]
     
d = {}

for i,v in enumerate(coins):
    if v != 0:
      d[i] = v      
l = []

for k,v in d.items():
    l.append(k)


'''
Debugger

komenda n - next line
komenda c - koncowy wynik

'''

import pdb

def add(L):
    
    size = len(L)
    total = 0
    iterator = 0
    pdb.set_trace()
    
    while iterator < size:
        
        total = total + L[iterator]
        
        iterator += 1
        
    return total

L = [1,2,3,4,5,6,7,'eight']
  
print(add(L))  
    
'''
Strings
'''

import math

pi = math.pi  
    
print(f'pi equal to {pi}')
  
print(f'pi equal to {pi:.3f}')  

print(f'pi equal to {pi: 10.3f}')
 
print(f'pi equal to {pi:^.3f}')  

print(f'pi equal to {pi:+.3f}') 
 
 """
Exercises
"""

'''
Question 1
Can you write a short program that will print out the version of Python
that you are using?
'''

import sys
print('Your version of Python is', sys.version)

'''
Question 2

Write a program that requests five names separated by commas and create a
list containing those names. Print your answer.
For example James,Alison,Fred,Sally,Matthew
should return ['James','Alison','Fred','Sally','Matthew']
'''

names = []

def imiona ():
    
    for i in range (1,6):
        
        imie = input('Wprowadz imie czlowieka numer {}: '.format(i))
        names.append(imie)

'''
Question 3
Write a program to determine whether a given number is within 10 of 100 or 200.
'''
def(number):
    if (abs(100 - number)) <= 10 or (abs(200 - number)) <= 10:
        return True
    else:
        return False
'''
Question 4
Write a program that takes a list of non-negative integers and prints each integer
to the screen the same number of times as the value of the integer, each new value
on a new line. For example
[2,3,4,1] would print:
22
333
4444
1
'''

def macierz(lista):
    
    for i in range(0,len(lista)):
        print(str(lista[i])*lista[i])
        
'''
Question 5
Write some code that will return the number of CPUs in the system.
'''

import multiprocessing

multiprocessing.cpu_count()   


'''
Question 6
Write a program that will return the sum of the digits of an integer.
'''

def liczba(x):
    
    y = [int(i) for i in str(x)]
    return sum(y)

'''
Question 8
Write a function that will check for the occurrence of double letters in
a string. If the string contains double letters next to each other it
will return True, otherwise it will return False.
'''

def slowo(x):
    
    x = str(x)
    
    for i in range(0,len(x)):
        
        if x[i] == x[i+1]:
            return True
    
    return False
            
     

import gc
gc.collect()

########## Additional exercises

string = 'Hey what is your name brother'

string.count('a')
string.find('what')
string.replace('what','elo')

list_1 = [1,2,3,'four','five','six']

del list_1[4]

list_1.remove('four')
list_1.append(2)
list_1.insert(5,'seven')
list_1.insert(0,5)

### list

list_2 = [23,41,2,4,5,1,67,89]

list_2.sort()
list_2.reverse()

print(min(list_2))

sum(list_2)

list_2.pop() ### usuwa ostatni element z listy

dodaj = [1,2,3]

list_2.extend(dodaj)

list_2.append(dodaj)
print(list_2)

list_2.index(67)
list_2.count(1)

4 in list_1


empty_list = [5]

if empty_list :
    print('lista jest pusta')
else:
    print('lista nie jest pusta')
    
## get method for dictionary
    

a = [1,2,3,4,5]
b= ['a','b','c','d','e']
    
new = dict(zip(a,b)) 
     
## funkcja get

# if (a,b) - jezeli a istnieje w liscie zwroci wartosc, jesli nie wyskoczy b
new.get(1,0) 
new.get(6,'do not exist')

          
# set w dictionary - jesli parametr nie wystepuje w liscie to wrzuc do dictionary

new.setdefault(9,'new element') 


slowo = 'asadsasdadbcbhfkdhjklashfkasfhksdjhfkwirdhksf'

slowo_2 = [i for i in slowo]    

slowo_3 = {key: slowo_2.count(key) for key in slowo_2}    

import matplotlib.pyplot as plt

%matplotlib inline

slowo_3.keys()
slowo_3.values()


plt.bar(slowo_3.keys(),slowo_3.values())


nowa = lambda x:x**2
nowa_2 = [nowa(x) for x in range(0,5)]

list_1.sort()
sorted(a)

## map function

x = [1,4,2,3,6]

y = map(lambda x:x**2,x)






















