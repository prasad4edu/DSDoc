a=10
b=20
b=50
#checking the datatypes
# by using the class function we can check the data types of each and variable
class(a)
x="datascience"
# if we mention object is quotes it will consider categorical value otherwise
# it will treat as variable or object
class(x)
# create the random numbers
x=rnorm(100)
y=rnorm(100)
z=rnorm(100)
# scatter plot will help us to compare the numeric varaibles
scatterplot3d(x,y,z)
# install a package
install.packages("scatterplot3d")
scatterplot3d(x,y,z)
# by using library function we can invoke the package into R core 
library(scatterplot3d)
scatterplot3d(x,y,z)
# discussion about vectors
a=20
abc="datascience"
class(a)
class(abc)
mode(a) # mode also tells about data type
mode(abc)
age=c(10,20,30,40,60,70)
age
age[1] # it will give you output 1st value of age varaible
age[3:5]
age[2]
age[-2]
#to calculate total positions,we are going to use length function
length(age)
age[6:8]
# adding the postion values
age[8]=100
age
# checking the data type of age
class(age)
age[7]="abc"
age
class(age)
# updating the values
age[1]=100
age
age[5]=30
age
age[7]=1000
age
class(age)
age1=as.numeric(age)
age1

age=c(10,20,40,70,100)
name=c("Tara","john","martin","nan","data")
english=c(80,85,90,45,68)
science=c(90,60,86,56,43)
# ls fucntion will tell us about list of objects
ls()
#creating a dataframe
student=data.frame(name,age,english,science)
View(student) # it will directly open the dataset
student
student[2,2]
student[4,3]
student[1,2:3]
student[1,-2]
student[4,-c(2,4)]
student[1] # it will print the first column in student table
student["name"]
age+5
age-10
age/2
age[6]=37
age/2
student
student+2
student
class(student$name)
class(student$age)
# to know the structure of a data, we have a one function i.e str
str(student)



wght=c(1,2,3,4,5,6,7)
wght[7]=0
wght
abc=data.frame(name,english,science,wght)
# creating the list data type
x=c(1:20)
y="datascience with R"
z=student
a=TRUE
mylist=list(x,y,a,z)
mylist
mylist[[1]][6]
student
# factor data type creation
gender=c("male","female","male","female","other")
gender1=factor(gender)
gender1
# creation of matrix
matrix(20,nrow = 5,ncol=4)
matrix(1:20,nrow = 5,ncol =4)
matrix(1:20,nrow=4,ncol = 5)
matrix(1:18,nrow = 4,ncol=5)

# rm fucntion, will remove the objects or tables
ls()
rm(gender) # removing the gender object from the global environment
x=-20
y=abs(x)
z=x+y
log(y)
exp(y)
id="CUst111213"
toupper(id) # to upper will convert all the strings into capital letters
tolower(id)
length(id)
substr(id,3,7)
substr(id,7,9)
grep("Ust",id)
grep(4,id)
grep(1,id)
class(id)

help(substr)
?substr
??substr
# importing the csv file into R
fiberbits=read.csv("C:\\Koti\\data science\\DS_batch1\\datasets\\fiberbits.csv")
library(readxl)




