
#######DataFrame############
import pandas as pd
data = {'name': ['Stan', 'Kyle', 'Eric', 'Kenny'],
        'age':[9, 9, 11, 12]}

df = pd.DataFrame(data)

######Importing the Datasets###########
#Data import from CSV files
import pandas as pd     # importing library pandas
Sales =pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Sales_sample.csv")
print(Sales)


#Data import from Excel files
wb_data = pd.read_excel("C:\\Koti\\data science\\DS_batch1\\datasets\\World Bank Indicators.xlsx" ,
                        "Data by country",
                        na_values=['NA'])
print(wb_data)

gnp =pd.read_sas("C:\\Koti\\data science\\DS_batch1\\datasets\\gnp.sas7bdat")
print(gnp)
Sales=sales
#############Basic commands on datasets#############
#Check list after Import 
# checking dimensions
Sales.shape
Sales.columns
list(Sales.columns)
Sales.columns.values
Sales.head(10)
Sales.unitsSold.head(10)
Sales["unitsSold"].head(10)
Sales.tail(5)
Sales.describe()
# describe function will work for entire dataset as well as particular variable also
Sales.unitsSold.describe()
# frequnecy count for the categorical variables
Sales.salesChannel.value_counts()
Sales.unitsSold.value_counts()
type(Sales)
Sales.dtypes
Sales.shape          # for dimension
Sales.columns.values # for variable names
Sales.columns
Sales.head(10)       # to print first 10 observations
Sales.tail(10)       # to print last 10 observations
Sales.dtypes         # to get the data type of each column

Sales.custId.dtypes   # to get data type of single variable
type(Sales.custId[1]) # gives datatype of the variable

Sales.describe()  #it gives summary about numeric variables in the dataset
Sales['custId'].describe() # give summary like no.of observations,mean,std,min,max,25,50,75 percentile

Sales.salesChannel.value_counts()  # we get frequency table for that variable
Sales.isnull()
Sales.info()
auto.info()
y=list(Sales.columns)
y
def abc(x):
    y=list(x.columns)
    z=sum(y.isnull())
    return z

abc(Sales)

Sales.custId.isnull()    # to check presence of missing values (NaN)
sum(Sales.custId.isnull()) # to get number of missing values
Sales.info()
AutoDatasetcsv.info()
Sales.sample(n=6) # Used for sampling the data, n is number of sample we want
Sales.sample(n=6,replace=True)

###################LAB:Basic Commands on Datasets#############################33
#Import “Superstore Sales  Data\\Sales_by_country_v1.csv” data
import pandas as pd
sales_country =pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Sales_by_country_v1.csv")

#How many rows and columns are there in this dataset?
sales_country.shape
#(998 rows, 7 columns)

#Print only column names in the dataset
sales_country.columns.values

#Print first 10 observations
sales_country.head(10)

#Print the last 5 observations
sales_country.tail(5)

#Get the summary of the dataset              ******
sales_country.describe()           # describe is close to summary() in R, but it wont give info about string variables 

#Print the structure of the data
sales_country.apply(lambda x: [x.unique()])  # this is close str() in R.
sales_country.apply(lambda x:[x.isnull()])
sales_country.isnull()

sales_country['custName'].unique()
                    
#Describe the field unitsSold, custCountry
sales_country.unitsSold.describe()

sales_country.custCountry.describe()       #describe wont give much info about string variable, so we are creating frequency table
sales_country.custCountry.value_counts()   #frequency table

#Create a new dataset by taking first 30 observations from this data
sales_new=sales_country.head(30)

#Print the resultant data
print(sales_new)

#Remove(delete) the new dataset
del(sales_new)


#############Manipulating the dataset################

#Sub setting the data#########################
import pandas as pd   


#Include encoding = "ISO-8859-1" or encoding = "utf8" to tackle the error
gdp=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\GDP.csv", encoding = "ISO-8859-1")
gdp.shape
gdp.columns.values

#New dataset with selected rows 
#Use iloc (index location)
gdp1 = gdp.head(10)
print(gdp1)

gdp2=gdp.iloc[[2,9,15,25]]
print(gdp2)

#New dataset by keeping selected columns
gdp3 = gdp[["Country", "Rank"]]
gdp3


#New dataset with selected rows and columns
gdp4 = gdp[["Country", "GDP"]][0:10]
gdp4

#New dataset with selected rows and excluding columns
gdp5=gdp.drop(["Country_code","GDP"], axis=1)
gdp5

######################################Lab: Sub setting the data#########################

#Data : "./Bank Marketing/bank_market.csv"
bank_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\bank_market.csv")
bank_data.shape
list(bank_data.columns)

#Select first 1000 rows only
bank_data1=bank_data.head(1000)
bank_data1.shape

#Select only four columns "Cust_num"  "age” "default" and  "balance"
bank_data2=bank_data[["Cust_num","age","default","balance"]]
bank_data2.head(5)

#Select 20,000 to 40,000 observations along with four variables  "Cust_num"  "job"       "marital" and   "education"

bank_data3=bank_data[["Cust_num","age","default","balance"]][20000:40000]
bank_data3.head(5)

#Select 5000 to 6000 observations drop  "poutcome“ and  "y"
bank_data4=bank_data.drop(['poutcome','y'], axis=1)[5000:6000]
bank_data4.head(5)

##################Subset with variable filter conditions#########################

#And condition & filters
bank_subset1=bank_data[(bank_data.age >50)]
bank_subset1.shape
bank_subset1=bank_data[(bank_data['age']> 50)]
bank_subset1.shape

bank_subset1=bank_data[(bank_data['age']>40) &  (bank_data['loan']=="no")]
bank_subset1

bank_data.shape
bank_subset1.shape

#OR condition & filters
bank_subset2=bank_data[(bank_data['age']>40) |  (bank_data['loan']=="no")]
bank_subset2

bank_data.shape
bank_subset2.shape

#AND, OR condition  Numeric and Character filters
bank_subset3= bank_data[(bank_data['age']>40) &  (bank_data['loan']=="no") | (bank_data['marital']=="single" )]
bank_subset3
bank_data.marital.value_counts()
import numpy as np
bank_data["marital1"]=np.where(bank_data.marital=="married","1",bank_data.marital)
bank_data["marital1"].value_counts()


##################Lab: Subset with variable filter conditions#########################

#Data : “./Automobile Data Set/AutoDataset.csv”
auto_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\AutoDataset.csv")
auto_data.shape
auto_data.columns.values

#Create a new dataset for exclusively Toyota cars
auto_data1=auto_data[auto_data[' make']=='toyota']
auto_data1

#Create a new dataset for all cars with city.mpg greater than 30 and engine size is less than 120
auto_data2=auto_data[(auto_data[' city-mpg']>30) & (auto_data[' engine-size']<120)]
auto_data2

#Create a new dataset by taking only  sedan cars. Keep only four variables(Make, body style, fuel type, price) in the final dataset
auto_data3=auto_data[auto_data[' body-style']=='sedan']
auto_data3

auto_data4=auto_data3[[' make',' body-style',' fuel-type',' price']]
auto_data4

#Create a new dataset by taking Audi, BMW or Porsche company makes. Drop two variables from the resultant dataset(price and normalized losses)
auto_data5=auto_data[(auto_data[' make']=='audi') | (auto_data[' make']=='bmw') | (auto_data[' make']=='porsche') ]
auto_data5

auto_data6=auto_data5.drop([' price',' normalized-losses'],axis=1)
auto_data6

##############################Calculated Fields###############################
auto_data.shape
auto_data.columns.values
auto_data['volume']=(auto_data[' length'])*(auto_data[' width'])*(auto_data[' height'])

auto_data.columns.values

auto_data['volume']

############################## Sorting the data###############################
#Its ascending by default

Online_Retail=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Online Retail.csv", encoding = "ISO-8859-1")
Online_Retail.columns.values

#Its ascending by default
Online_Retail_sort=Online_Retail.sort('UnitPrice')
Online_Retail_sort.head(20)

#Use ascending=False for descending
Online_Retail_sort=Online_Retail.sort('UnitPrice',ascending=False)
Online_Retail_sort.head(20)


##############################LAB: Sorting the data###############################

#AutoDataset
auto_data.shape
auto_data.columns.values

#Sort the dataset based on length
auto_sort=auto_data.sort(' length')  #default sorted in ascending order
auto_sort.head(10)

auto_sort=auto_data.sort(' length',ascending=False)  # sorted in descending order
auto_sort.head(10)

##################################Identifying Duplicates###############################
bill_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Telecom Data Analysis\\Bill.csv")
bill_data.head(10)
bill_data.shape
bill_data.columns
#Ide`ntify duplicates records in the data
dupes=bill_data.duplicated()
sum(dupes)

#Removing Duplicates
bill_data.shape
bill_data_uniq=bill_data.drop_duplicates()
bill_data_uniq.shape

#Identify duplicates in complaints data based on cust_id
dupe_id=bill_data.cust_id.duplicated()
sum(dupe_id)
#Removing duplicates based on a variable
bill_data.shape
bill_data_cust_uniq=bill_data.drop_duplicates(['cust_id'])
bill_data_cust_uniq.shape

##################################LAB: Handling Duplicates in R###############################

#DataSet: "./Telecom Data Analysis/Complaints.csv"
comp_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Telecom Data Analysis\\Complaints.csv")
comp_data.shape
comp_data.columns.values

#Identify overall duplicates in complaints data
dupe=comp_data.duplicated()
sum(dupe) # gives total number of duplicates in data

#Create a new dataset by removing overall duplicates in Complaints data
comp_data1=comp_data.drop_duplicates()

#Identify duplicates in complaints data based on cust_id
dupe_id=comp_data.cust_id.duplicated()
sum(dupe_id)
#Create a new dataset by removing duplicates based on cust_id in Complaints data
comp_data2=comp_data.drop_duplicates(['cust_id'])

################################## Data Joins#################################################

orders=pd.read_csv("C:\\Koti\\data science\\DS_batch1\datasets\\TV Commercial Slots Analysis\\orders.csv")
orders.shape
orders.head(10)

slots=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\TV Commercial Slots Analysis\\slots.csv")
slots.shape

#Python duplicates based on cust_id
sum(orders.Unique_id.duplicated())
sum(slots.Unique_id.duplicated())

orders1=orders.drop_duplicates(['Unique_id'])
slots1=slots.drop_duplicates(['Unique_id'])

sum(orders1.Unique_id.duplicated())
sum(slots1.Unique_id.duplicated())

###Inner Join

inner_data=pd.merge(orders1, slots1, on='Unique_id', how='inner')
inner_data.shape

###Outer Join
outer_data=pd.merge(orders1, slots1, on='Unique_id', how='outer')
outer_data.shape

##Left outer Join
L_outer_data=pd.merge(orders1, slots1, on='Unique_id', how='left')
L_outer_data.shape

###Righ outer Join
R_outer_data=pd.merge(orders1, slots1, on='Unique_id', how='right')
R_outer_data.shape

##################################LAB: Data Joins#################################################

#Datasets “./Telecom Data Analysis/Bill.csv” and “./Telecom Data Analysis/Complaints.csv”
comp_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Telecom Data Analysis\\Complaints.csv")
comp_data.shape
comp_data.columns.values

bill_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Telecom Data Analysis\\Bill.csv")
bill_data.shape
bill_data.columns.values

#Import the data and remove duplicates based on cust_id
comp_data1=comp_data.drop_duplicates(['cust_id'])
comp_data1.shape

bill_data1=bill_data.drop_duplicates(['cust_id'])
bill_data1.shape

#All the customers who appear either in bill data or complaints data 
combined1=pd.merge(comp_data1, bill_data1, on='cust_id', how='outer')
combined1.shape

#All the customers who appear both in bill data and complaints data
combined2=pd.merge(comp_data1, bill_data1, on='cust_id', how='inner')
combined2.shape

#All the customers from bill data: Customers who have bill data along with their complaints
combined3=pd.merge(comp_data1, bill_data1, on='cust_id', how='right')
combined3.shape

#All the customers from complaints data: Customers who have Complaints data along with their bill info
combined4=pd.merge(comp_data1, bill_data1, on='cust_id', how='left')
combined4.shape


#############################
###################
##Exporting to an external file
## we will export the left join output dataframe into a csv file
L_outer_data.to_csv('d:\\left_join.csv')
