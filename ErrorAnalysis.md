# Error Analysis in Machine Learning
====================================

## Optimization and Generalization

One of the fundamental issue in machine learning is the balance between optimization and generalization. *Optimization* refers to the process of adjusting a model to get the best performance possible on the training data, while *Generalization* refers to how well the trained model performs on data it has never seen before. The ultimate goal of a learning system is to come up with a good generalization.

While training a model at first, the optimization and generalization are correlated; i.e., the lower the loss on training data, the lower the loss on the test data. As the process continues, the model is said to be __underfit__ meaning there is still progress to be made and the model has not yet learned all relevant patterns in the training data. But, after certain number of iterations over the training data, `generalization` stops improving and validation metrics gets halted and then begin to degrade, which means the model is starting to __overfit__ indicating that it's starting to learn patterns which are specific to the training data but which are misleading or irrelevant when it comes to new data.

## Overfitting and Underfitting

*Overfitting* and *Underfitting* are common problems in machine learning. They are a direct outcome of the model's generalization and are a phenomenon which occurs when the learned model fails to perform well on the new or real data.

- Overfitting

__Overfitting__ is quite a common problem where a model performs well on the training data but does not generalize well to unseen (test or new) data. If a model suffers from overfitting, the model will have high Variance, which can be caused by having too many parameters leading to a rather too complex model.


- Underfitting

A model can suffer from __Underfitting__ when the model is not complex enough to capture the pattern in the training data well and therefore cannot perform well on unseen data.

The below figure provides an idea of how the data fits on a model for some random data set.

![curve fitting](images/fitting.png "Illustrating underfitting and overfitting")

## Bias versus Variance

There are two common categories of error(s) in the world of machine intelligence, the training error and the validation error. These two can come up from the data set distributed as training and validation(test) data set(s).

Training error is the difference between the known correct output for the inputs and the actual output of the prediction model. During the course of training the training error is reduced until the model produces an accurate (or a near accurate) prediction for the training set. 

Validation error is the difference between the known correct output for the validation set and the actual output of the prediction model.

The errors in the prediction can be decomposed into two main sub-components:

- Error due to __Bias__
- Error due to __Variance__

Bias and Variance are two statistical concepts which are important for almost all types of machine learning algorithms

- Bias:

Bias can be considered as an error from erroneous assumption in the learning algorithm or model. High *Bias* means the model is not *fitting* well on the training set. This means the training error will be large. Low *Bias* means that the model is fitting well, and the training error will be low.

- Variance:

Variance is the amount that the estimate of the target function will change if different training data was used.High *Variance* means that the model is not able to make accurate predictions on the test or validation set and the validation error will be large. Low *Variance* means the model is successful in breaking out of it's training data.

__High Bias is equivalent to aiming at the wrong place, while High Variance is equivalent to aiming an unsteady target__

Here is an illustration of the Bias vs Variance using the analogy of archery.

![bias variance tradeoff](images/bias-variance.png "The Bias Variance trade-off")

### Bias versus Variance trade-off

Every model has both bias and variance error components along with some noise. **Bias** and **Variance** are inversely related to each other, while trying to reduce one component, the other component of the model will increase. One needs to balance both in order to create a good fit. The ideal model will have both __low bias__ and __low variance__.


