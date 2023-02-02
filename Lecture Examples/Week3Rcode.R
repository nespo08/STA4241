## read in the data
Input=("
       Flight Temp Failure
       1 66 0 
       2 70 1 
       3 69 0 
       4 68 0 
       5 67 0 
       6 72 0 
       7 73 0 
       8 70 0 
       9 57 1 
       10 63 1 
       11 70 1 
       12 78 0 
       13 67 0 
       14 53 1 
       15 67 0 
       16 75 0
       17 70 0
       18 81 0
       19 76 0
       20 79 0
       21 75 1
       22 76 0
       23 58 1
       ")
preC=read.table(textConnection(Input),header=TRUE)

## Plot data
plot(preC$Temp, preC$Failure, xlab="Temperature", ylab="Failure")

# Logit model
preC.logit=glm(Failure~Temp,family=binomial(link=logit),data=preC)
summary(preC.logit)

## Can use curve function
curve(exp(preC.logit$coefficients[1]+preC.logit$coefficients[2]*x)/
        (1+exp(preC.logit$coefficients[1]+preC.logit$coefficients[2]*x)),	
      from=50,to=85,col=2,lty=2,lwd=2,add=TRUE)

## or can do it manually
s = seq(50, 85, length=1000)
logitProbs = predict(preC.logit, newdata=data.frame(Temp = s),
                     type = "response")
lines(s, logitProbs, lty=2, col=2, lwd=2)

## Predict at challenger temperature
predict.glm(preC.logit,newdata=data.frame(Temp=36),type="response")

## probit model
preC.probit=glm(Failure~Temp,family=binomial(link=probit),data=preC)
curve(pnorm(preC.probit$coefficients[1]+preC.probit$coefficients[2]*x),	
      from=50,to=85,col=3,lty=2,lwd=2,add=TRUE)

## Linear model
preC.linear = lm(Failure ~ Temp, data=preC)
curve(preC.linear$coefficients[1]+preC.linear$coefficients[2]*x,	
      from=50,to=85,col=1,lty=2,lwd=2,add=TRUE)
legend("topright", c("Linear", "Logit", "Probit"), col=1:3, lwd=2, lty=2)

