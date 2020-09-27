# Example 2.1
# get/set working directory
getwd()
setwd("C:/Course17/ceR/R")

# a few peculiar syntax: <-, ~, #
# get help
str(help)
help()
help(lm)
help.search("regression")

# install and use packages
# AER, Ecdat, car, ...
library(help="datasets")

# Let's Begin

A <- matrix(c(1,2,3,0,1,4,0,0,1), nrow = 3, byrow = TRUE)
B=c(2,7,1)
C <- matrix(B, nrow = 3)
C1 <- matrix(B, nrow = 3,ncol = 3)
C2 <- matrix(B, nrow = 3,ncol = 3,byrow = TRUE)

print ("Matrix A")
A
print ("Matrix C") 
C
print ("A*C")
A%*%C
print ("A.*C")
A*C1
print ("A.*C'") 
A*C2

rm(list=ls())
