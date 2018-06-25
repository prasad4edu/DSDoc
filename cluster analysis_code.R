# Cluster analysis in R
#importing the data into R
library(readr)
cars<- read_csv("G:/DS_batch1/R/car_cluster.csv")
View(cars)
# to check the dimensions of car data
dim(cars)
head(cars)
# to standardize the variables
cars.use = cars[,-c(1,2)]
# calcuating the median for all numerical values
medians = apply(cars.use,2,median)
# calculating the median absolute deviation 
mads = apply(cars.use,2,mad)
# standardize the variables 
cars.use = scale(cars.use,center=medians,scale=mads)
cars.use
fix(cars.use)
# calcualting the distance measure from attribute to other attributs
cars.dist = dist(cars.use)
fix(cars.dist)
#performing hierarchial clustering
cars.hclust = hclust(cars.dist)
# plot the clusters 
plot(cars.hclust,labels=cars$Car,main='Default from hclust')
# cluster interpretation
groups.3 = cutree(cars.hclust,3)
table(groups.3)
# To look at the cluster parts of analysis from 2 to 6
counts = sapply(2:6,function(ncl)table(cutree(cars.hclust,ncl)))
names(counts) = 2:6
counts
# to check which clusters are in each group
cars$Car[groups.3 == 3]
# if we want to do the same thing for all the groups at once, we can use sapply:
sapply(unique(groups.3),function(g)cars$Car[groups.3 == g])
# how cluster performs based on country
table(groups.3,cars$Country)
# calculating the median for cluster data
aggregate(cars.use,list(groups.3),median)
# calculating the median on original data
aggregate(cars[,-c(1,2)],list(groups.3),median)

# PAM: Partitioning Around Medoids
library(cluster)
cars.pam = pam(cars.dist,3)
# pam considers one cluster at a time
names(cars.pam)
# compare the hclust and pam method
table(groups.3,cars.pam$clustering)
#The solutions seem to agree, except for 1 observations that hclust put in group 2 and pam put in group 3. Which observations was it?
cars$Car[groups.3 != cars.pam$clustering]
# plot the clusters
plot(cars.pam)
#Range of SC	Interpretation
#0.71-1.0	A strong structure has been found
#0.51-0.70	A reasonable structure has been found
#0.26-0.50	The structure is weak and could be artificial
#< 0.25	No substantial structure has been found

# K-means clustering

library(datasets)
fix(iris)
head(iris)
table(iris$Species)
library(ggplot2)
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()
set.seed(20)
irisCluster <- kmeans(iris[, c(2,4)], 3, nstart = 20)
irisCluster
table(irisCluster$cluster, iris$Species)
irisCluster$cluster <- as.factor(irisCluster$cluster)
ggplot(iris, aes(Petal.Length, Petal.Width, color = irisCluster$cluster)) + geom_point()
