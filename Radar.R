dim(FDSR)
names(FDSR)
library(rpart)
#Building Tree Model
Radar_Tree<-rpart(AccIncTypeID~., method="class", data=FDSR, control=rpart.control(minsplit=30))
Radar_Tree

#Plotting the trees
plot(Radar_Tree, uniform=TRUE)
text(Radar_Tree, use.n=TRUE, all=TRUE)

#A better looking tree
install.packages("rattle") 
library(rattle)
fancyRpartPlot(Radar_Tree,palettes=c("Greys", "Oranges"))

