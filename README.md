# GradientDescent

## Overview/Summary

This is the Implementation of gradient descent algorithm to find the value of parameters(m,b) so that the cost function is minimized i.e. finding the line of best fit on the data.

### Visualize data

Visualizing the data

![Data](https://github.com/ketanpandey01/GradientDescent/blob/master/resources/visualizingData.PNG)


### Cost Function - Sum of Squared Error

A simple SSE algorithm for measuring error

```
def CostFunction(m,b,data):
    sumError = 0
    for itr in range(m_examples):
        feature = data[itr,0]
        label = data[itr,1]
        predLabel = (m * feature) + b
        sumError += (label - predLabel)**2
    sumError = sumError/m_examples
    return sumError
```
### Visualize data plus fitting line

* Before fitting the parameters

![Data plus fitting line](https://github.com/ketanpandey01/GradientDescent/blob/master/resources/errorFit.PNG)

* After Running Gradient Descent

![Data plus fitting line](https://github.com/ketanpandey01/GradientDescent/blob/master/resources/bestFit.PNG)

### Plot - Error

The error changing as we move toward the minimum.

![Error](https://github.com/ketanpandey01/GradientDescent/blob/master/resources/visualizingError.PNG)


## Dependencies]

* numpy
* pandas (read the dataset)
* matplotlib (plotting)


