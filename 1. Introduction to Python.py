##########################################
#Basic commands
571+95
19*17
print(57+39)
print(19*17)
print("Data Vedi")
Print("Floris")
#Division example
34/56

#Error
Print(600+900) #used Print() instead of print()
576-'96'

#LAB: Basic Commands
973*75
22/7

##########################################
#Assignment of variables
income=12000
income
print(income)
x=20
x

y=30
z=x*y
z

del x #deletes the variable
x

name="Jack"
name

print(name)


#Is there a difference between output of name and print(name)?

book_name="Practical business analytics \n using SAS"
book_name

print(book_name)


##Variables are lost after restarting shell

##########################################
#Naming convention

1x=20 #Doesn't work

x1=20 #works
x1

x.1=20 #Doesn't work
x.1

x_1=20 #works
x_1


##Variable and Datatypes
x=89
x
type(x)

y=2.98
y
type(y)

#delete variables
del book_name
del x1 
del x

##########################################
#Type of Objects 

##Numbers
age=30.00
age

weight=102.88
weight

x=17
x**2 #Square of x
y=-10
type(y)
##Defining Strings 
name="Sheldon"
msg=" Data Science Classes"

name
msg
name[0:10]
#Accessing strings
print(name[0])
print(name[1])

print(msg[0:9]) #This is as good as substring

len(msg) #length of string
print(msg[10:len(msg)])

#Displaying string multiple time
msg="Site under Construction"
msg*10
msg*50

#There is a difference between print and just displaying a variable
message="Data Science on R and Data Science on Python \n"
message*10
print(message*10)

#String Concatenation
msg1="Site under Construction "
msg2=msg1+"Go to home page \n"
print(msg2)
print(msg2*10)

#List is a hybrid datatype
#Similar to array, but all the elements need not be of sametype

#Creating a list
mylist1=['Sheldon','Male', 25]

#Accessing list elements
mylist1[0] #Python indexing starts from 0
mylist1[1]
mylist1[-3]

#Appending to a list
mylist2=['L.A','No 173', "CR108877"]
mylist2[3]
final_list=mylist1+mylist2
final_list

#Updating list elements
final_list[2]
final_list[2]=35
final_list[2:4]=[45,"SSS","kkkk"]

#Length of list
len(final_list)

#Deleting an element in list
del final_list[5]
final_list


#Tuples
my_tuple=('Mark','Male', 55)
my_tuple
my_tuple[1]
my_tuple[2]

my_tuple[2]*10
print(my_tuple[0]*10)
#tuple can't be updated
my_list=['Sheldon','Male', 25]
my_tuple=('Mark','M', 55)

my_list[2]=30
my_list

my_tuple[2]=40

#Dictionaries in Python
#The key value pairs

city={0:"LA", 1:"PA" , 2:"FL"}
city

#Accessing values 
city[0]
city[1]
city[2]

#Make sure that we give the right key index
names={1:"David", 6:"Bill", 9:"Jim"}
names
names[0] #Doesn't work, why?
names[1]
names[2]
names[6]
names[9]

#Key need not be a number always
edu={"David":"Bsc", "Bill":"Msc", "Jim":"Phd"}
edu

edu[0]
edu[1]
edu[David]
edu["David"]

#Updating values in dictionary 
edu
edu["David"]
edu["David"]="Mtech"
edu["David"]
edu

#Updating keys in dictionary 
#Delete the key and value element first and then add new element

city={0:"LA", 1:"PA" , 2:"FL"}

#How to male 6 as "LA"

del city[0]
city

city[6]="LA"
city

#Looking at keys and values
city.keys()
city.values()

edu.keys()
edu.values()


##########################################
#If-Then-Else statement
#Checks the condition is true or false and do operations accordingly 

#If Condition
age=30
if age<50:
    print("Group1")

print("Done with If")


##If else statement

age=50
if age<50:
    print("Group1")
else:
    print("Group2")
print("Done with If else")


#Multiple conditions in if 
#Use elif

marks=20

if(marks<30):
    print("fail")
elif(marks<60):
    print("Second Class")
elif(marks<80):
     print("First Class")
elif(marks<100):
     print("Distinction")
else:
    print("Error in Marks")
    

marks=120

if(marks<30):
    print("fail")
elif(marks<60):
    print("Second Class")
elif(marks<80):
     print("First Class")
elif(marks<100):
     print("Distinction")
else:
    print("Error in Marks")
    
    

#Nested if

x=35

if(x<50):
    print("Number is less than 50")
    if(x<40):
         print ("Number is less than 40")
         if(x<30):
             print("Number is less than 30")
         else:
             print("Number is greater than 30")
    else:
        print("Number is greater than 40")
else:
    print("Number is greater than 50")


x=45

if(x<50):
    print("Number is less than 50")
    if(x<40):
         print ("Number is less than 40")
         if(x<30):
             print("Number is less than 30")
         else:
             print("Number is greater than 30")
    else:
        print("Number is greater than 40")
else:
    print("Number is greater than 50")


##########################################
##For loop

#Example-1

my_num=1
for i in range(1,20):
    my_num=my_num+1
    print("my num value is", my_num)

#Example-2

sumx = 0
for x in range(1,20): 
     sumx = sumx + x
     print(sumx)

##The break statement 
#To stop execution of a loop
#Stopping the loop in midway

sumx = 0 
for x in range(1,200): 
     sumx = sumx + x
     if(sumx>500):
         break
     print(sumx)
     
##########################################
#functions

def abc(a):
    c = a*a
    return c
abc(10)

###
def divide(var1, var2):
	a = var1/var2
	print(a)
divide(20,10)

##################################################
#Packages

log(10)
exp(5)
sqrt(256)

import math 
math.log(10)
math.exp(5)
math.sqrt(256)

import math as mt
mt.log(10)
mt.exp(5)
mt.sqrt(256)

#numpy package

import numpy as np

income = np.array([9000, 8500, 9800, 12000, 7900, 6700, 10000])
print(income) 
print(income[0])
income[0]

expenses=income*0.65
print(expenses)

savings=income-expenses
print(savings)


#pandas package
import pandas as pd
buyer_profile = pd.read_csv('d:\\Datasets\\Buyers Profiles\\Train_data.csv')

print(buyer_profile)

buyer_profile.Age
buyer_profile.Bought

buyer_profile.Age[0]
buyer_profile.Age[0:10]

#Matplotlib

import matplotlib as mp
import numpy as np

X = np.random.normal(0,1,1000)
Y = np.random.normal(0,1,1000)

mp.pyplot.scatter(X,Y)

#Sklearn 
import pandas as pd

air_passengers = pd.read_csv('d:\\statinfer\\Datasets\\AirPassengers\\AirPassengers.csv')
air_passengers

x=air['Promotion_Budget']
x[1]
x=x.reshape(-1,1)
x

y=air_passengers['Passengers']
y
y=y.reshape(-1,1)
y

from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(x, y)
print('Coefficients: \n', reg.coef_)

















