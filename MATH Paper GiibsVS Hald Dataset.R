# The experiment is to execute Bayesian Variable Selection for linear regression models using Gibbssampling
# on the Hald data set referenced in paper "Variable Selection Via Gibbs Sampling" by George and McCulloch
# http://www.snn.ru.nl/~bertk/machinelearning/2290777.pdf
# We will then compare and contrast with traditional linear reqression (adjusted r-squared) and stepwise subset selection


# References
# http://www.snn.ru.nl/~bertk/machinelearning/2290777.pdf
# https://cran.r-project.org/web/packages/BayesVarSel/BayesVarSel.pdf
# https://rdrr.io/cran/BayesVarSel/man/Bvs.html
# http://blog.uclm.es/gonzalogarciadonato/files/2016/11/articleJSS_v2.pdf-english-revised-by-Neil4.pdf

# install required packages
# install.packages("ridge")
# install.packages("BayesVarSel")
# install.packages("BVS")
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("leaps")



# Load required libraries
library(ridge)
library(BayesVarSel)
library(BVS)
library(tidyverse)
library(caret)
library(leaps)

# --------------------------------------------------------------------------------- #
# ------------------------ Initial Experiment on Hald Data ------------------------ #
# --------------------------------------------------------------------------------- #

# Load Hald Dataset
data(Hald)

# ----- Comments ---------------------#
# Variable selection using gibbs sampling
# (Zellner) g-prior for
# regression parameters and constant prior
# over the model space
# In this Gibbs sampling scheme, we perform 10100 iterations,
# of which the first 100 are discharged (burnin) and of the remaining
# only one each 10 is kept.
# as initial model we use the Full model

hald.GibbsBvs.fullmodel<- GibbsBvs(formula= y ~ ., data=Hald, prior.betas="gZellner",
                         prior.models="Constant", n.iter=10000, init.model="Full", n.burnin=100,
                         time.test = FALSE)

summary(hald.GibbsBvs.fullmodel)

# ----- Console Output ---------------------#
#Inclusion Probabilities:
#    Incl.prob. HPM MPM
#x1      0.893   *   *
#x2     0.6234   *   *
#x3     0.3466        
#x4     0.5828       *
#  ---
#  Code: HPM stands for Highest posterior Probability Model and
#MPM for Median Probability Model.
#Results are estimates based on the visited models.

# Reduced model
hald.GibbsBvs.reducedmodel<- GibbsBvs(formula= y ~ x1+x2, data=Hald, prior.betas="gZellner",
                                   prior.models="Constant", n.iter=10000, init.model="Full", n.burnin=100,
                                   time.test = FALSE)

summary(hald.GibbsBvs.reducedmodel)

# ----- Console Output ---------------------#
#Inclusion Probabilities:
#     Incl.prob. HPM MPM
#x1     0.9992   *   *
#x2     0.9998   *   *
#  ---
#  Code: HPM stands for Highest posterior Probability Model and
#MPM for Median Probability Model.
#Results are estimates based on the visited models.

# Using Bayesian Variable Selection for Linear regression mmodels (Bvs) to validate and compare results

hald_Bvs <- Bvs(formula = y ~ x1 + x2 + x3 + x4, data = Hald)

summary(hald_Bvs)

# ----- Console Output ---------------------#
#Info. . . .
#Most complex model has 5 covariates
#From those 1 is fixed and we should select from the remaining 4 
#x1, x2, x3, x4
#The problem has a total of 16 competing models
#Of these, the  10 most probable (a posteriori) are kept
#Working on the problem...please wait.

#Inclusion Probabilities:
#    Incl.prob. HPM MPM
#x1     0.9762   *   *
#x2     0.7563   *   *
#x3     0.2624        
#x4     0.4153        
#---
#  Code: HPM stands for Highest posterior Probability Model and
#MPM for Median Probability Model.

# ----- Comments ---------------------#
# Simple linear regression and r-squared for comparison

lm.Hald.fullmodel = lm(y~., Hald)
summary(lm.Hald.fullmodel)

# ----- Console Output ---------------------#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)  
#(Intercept)  62.4054    70.0710   0.891   0.3991  
#x1            1.5511     0.7448   2.083   0.0708 .
#x2            0.5102     0.7238   0.705   0.5009  
#x3            0.1019     0.7547   0.135   0.8959  
#x4           -0.1441     0.7091  -0.203   0.8441  
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 2.446 on 8 degrees of freedom
#Multiple R-squared:  0.9824,	Adjusted R-squared:  0.9736 
#F-statistic: 111.5 on 4 and 8 DF,  p-value: 4.756e-07

lm.Hald.reducemodel = lm(y~x1+x2, Hald)
summary(lm.Hald.reducemodel)

# ----- Console Output ---------------------#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept) 52.57735    2.28617   23.00 5.46e-10 ***
#  x1           1.46831    0.12130   12.11 2.69e-07 ***
#  x2           0.66225    0.04585   14.44 5.03e-08 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 2.406 on 10 degrees of freedom
#Multiple R-squared:  0.9787,	Adjusted R-squared:  0.9744 
#F-statistic: 229.5 on 2 and 10 DF,  p-value: 4.407e-09


# ----- Comments ---------------------#
# Stepwise selection on  a linear model for comparison
# Set seed for reproducibility
set.seed(123)
# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
# Train the model
step.model <- train(y~x1+x2+x3+x4, data = Hald,
                    method = "leapSeq", 
                    tuneGrid = data.frame(nvmax = 1:2),
                    trControl = train.control
)

summary(step.model$finalModel)

# ----- Console Output ---------------------#
#Subset selection object
#4 Variables  (and intercept)
#Forced in Forced out
#x1     FALSE      FALSE
#x2     FALSE      FALSE
#x3     FALSE      FALSE
#x4     FALSE      FALSE
#1 subsets of each size up to 2
#Selection Algorithm: 'sequential replacement'
#          x1  x2  x3  x4 
#1  ( 1 ) " " " " " " "*"
#2  ( 1 ) "*" "*" " " " "
# -----#
plot(lm.Hald.fullmodel, option="conditional")
plot(lm.Hald.reducemodel, option="conditional")
