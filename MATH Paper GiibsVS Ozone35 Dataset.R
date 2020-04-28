# The experiment is to execute Bayesian Variable Selection for linear regression models using Gibbs sampling
# on the Ozone35 data set referenced in "Casella, G. and Moreno, E. (2006)<DOI:10.1198/016214505000000646> Objective Bayesian variable selection. Journal of the American Statistical Association, 101(473).
# https://www.stat.washington.edu/courses/stat527/s14/readings/CasellaMoreno_JASA_2006.pdf
# The data set has 35 variables



# References
# https://www.stat.washington.edu/courses/stat527/s14/readings/CasellaMoreno_JASA_2006.pdf
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
# install.packages("eqs2lavaan")



# Load required libraries
library(ridge)
library(BayesVarSel)
library(BVS)
library(tidyverse)
library(caret)
library(leaps)
library(glmnet)
library(corrplot)
library(dplyr)
library(RColorBrewer)
library(datasets)
library(eqs2lavaan)

# --------------------------------------------------------------------------------- #
# ------------------------ Second Experiment on Ozone35 Data ------------------------ #
# --------------------------------------------------------------------------------- #

# Load Hald Dataset
data(Ozone35)
set.seed(123)
Ozone35_orig = Ozone35
split <- round(nrow(Ozone35_orig) * 0.80)
Ozone35_train = Ozone35_orig[1:split,]
Ozone35_test = Ozone35_orig[(split + 1):nrow(Ozone35_orig), ]
nrow(Ozone35_orig)
nrow(Ozone35_train)
nrow(Ozone35_test)

# correlation plot
M <-cor(Ozone35_train)
corrplot(M, type="upper", order="hclust", col=brewer.pal(n=8, name="RdYlBu"))

#5-fold cross-validated model:
train.control <- trainControl(method = "cv", number = 5)
Ozone35CV <- train(y ~ ., data=Ozone35_train, method="lm", trControl=train.control)


finModel <- Ozone35CV$finalModel
finModel
predicts <- predict(finModel, Ozone35_test)
summary(finModel)
Ozone35_testcv = Ozone35_test
Ozone35_testcv$fitted_values = predicts
mean((Ozone35_testcv$y - Ozone35_testcv$fitted_values)^2)
#  26.01848


# lasso regression
x = model.matrix(y~., Ozone35_train)
y = Ozone35_train$y
set.seed(5678)
lambda <- 10^seq(10, -2, length = 100)
lasso.mod <- glmnet(x, y, alpha = 1, lambda = lambda)
cv.out <- cv.glmnet(x, y, alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
#0.02236135

newx = model.matrix(y~., Ozone35_test)
lasso.pred = predict(lasso.mod, s = bestlam, newx)
mean((lasso.pred-Ozone35_test$y)^2)
# 25.18038
lasso.coef  = predict(lasso.mod, type = 'coefficients', s = bestlam)
lasso.coef
# Eliminates 13 Variables, 22 still remain
# concole output
#(Intercept) -2.175134e+01
#(Intercept)  .           
#x4           .           
#x5           .           
#x6           .           
#x7           .           
#x8           1.110815e-03
#x9           1.625016e-02
#x10          .           
#x4.x4        7.547358e-07
#x4.x5        .           
#x4.x6        1.528497e-05
#x4.x7        .           
#x4.x8        .           
#x4.x9        .           
#x4.x10       .           
#x5.x5       -1.036284e-01
#x5.x6        .           
#x5.x7       -8.260029e-04
#x5.x8        .           
#x5.x9        1.327944e-02
#x5.x10       4.243334e-03
#x6.x6       -1.532618e-03
#x6.x7        3.122803e-03
#x6.x8       -2.340310e-05
#x6.x9        1.790178e-05
#x6.x10       6.912824e-05
#x7.x7        1.626493e-03
#x7.x8       -1.027678e-05
#x7.x9       -1.127902e-03
#x7.x10      -9.327268e-04
#x8.x8        .           
#x8.x9        2.172081e-06
#x8.x10       4.829100e-07
#x9.x9       -6.914419e-04
#x9.x10      -3.203076e-05
#x10.x10      4.012404e-05


# Using Gibbs
Ozone35.GibbsBvs.fullmodel<- GibbsBvs(formula= y ~ ., data=Ozone35_train, prior.betas="gZellner",
                                   prior.models="Constant", n.iter=10000, init.model="Full", n.burnin=100,
                                   time.test = FALSE)

summary(Ozone35.GibbsBvs.fullmodel)
# Console Output
#Inclusion Probabilities:
#            Incl.prob.  HPM   MPM
#x4          0.1762        
#x5          0.1581        
#x6          0.2913        
#x7           0.244        
#x8          0.2192        
#x9           0.236        
#x10           0.32      *    
#x4.x4       0.1726        
#x4.x5       0.1694        
#x4.x6       0.3171      *    
#x4.x7       0.3398      *    
#x4.x8        0.234        
#x4.x9       0.2545        
#x4.x10      0.3535        
#x5.x5       0.2386        
#x5.x6       0.1423        
#x5.x7       0.1639        
#x5.x8       0.1394        
#x5.x9       0.1959        
#x5.x10      0.2037        
#x6.x6       0.5369            *
#x6.x7       0.5771            *
#x6.x8        0.596       *    *
#x6.x9       0.1443        
#x6.x10       0.135        
#x7.x7       0.2443        
#x7.x8       0.3684        
#x7.x9       0.3711        
#x7.x10      0.7776       *    *
#x8.x8       0.1298        
#x8.x9       0.1792        
#x8.x10       0.177        
#x9.x9       0.5838            *
#x9.x10      0.1118        
#x10.x10     0.1565        
#---
#  Code: HPM stands for Highest posterior Probability Model and
#MPM for Median Probability Model.
#Results are estimates based on the visited models.
# eliminitaes 30 variables, 5 remain

# Reduced Model
Ozone35.GibbsBvs.reducedmodel<- GibbsBvs(formula= y ~ x10+x4.x6+x4.x7+x6.x8+x7.x10, data=Ozone35_train, prior.betas="gZellner",
                                      prior.models="Constant", n.iter=10000, init.model="Full", n.burnin=100,
                                      time.test = FALSE)

summary(Ozone35.GibbsBvs.reducedmodel)

# Console Outout

#Inclusion Probabilities:
#            Incl.prob. HPM MPM
#x10        0.9941       *   *
#x4.x6      0.9938       *   *
#x4.x7           1       *   *
#x6.x8       0.999       *   *
#x7.x10     0.9979       *   *
#  ---
#  Code: HPM stands for Highest posterior Probability Model and
#MPM for Median Probability Model.
#Results are estimates based on the visited models.

# Linear regression with reduced coeff.

lm.Ozone35.reducemodel = lm(y ~ x10+x4.x6+x4.x7+x6.x8+x7.x10, Ozone35_train)
summary(lm.Ozone35.reducemodel)

# Console Output

# Residuals:
#  Min       1Q   Median       3Q      Max 
#-10.5613  -2.7565  -0.2178   2.7089  12.9488 


#               Estimate Std. Error     t value       Pr(>|t|)    
#(Intercept)   -1.823e+01  3.155e+00    -5.778      4.92e-08 ***
#  x10          9.811e-02  2.343e-02    4.187      5.04e-05 ***
#  x4.x6        1.889e-05  4.690e-06    4.029      9.28e-05 ***
#  x4.x7        8.016e-05  8.555e-06    9.371      < 2e-16 ***
#  x6.x8       -1.894e-05  4.037e-06    -4.691      6.54e-06 ***
#  x7.x10      -1.949e-03  4.141e-04    -4.706      6.15e-06 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 4.247 on 136 degrees of freedom
#Multiple R-squared:  0.7477,	Adjusted R-squared:  0.7384 
#F-statistic:  80.6 on 5 and 136 DF,  p-value: < 2.2e-16

lm.Ozone35.reducemodel.predictions = predict(lm.Ozone35.reducemodel, Ozone35_test)
lm.Ozone35.reducemodel.predictions
mean((lm.Ozone35.reducemodel.predictions-Ozone35_test$y)^2)
#18.58677


