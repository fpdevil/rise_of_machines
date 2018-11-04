# Logistic Regression in Machine Learning

Logistic Regression (sometimes called as Logistic model or Logit model) is a multivariable method for analyzing the relationship between multiple independent variables and a categorical dependent variable. Here multivariate or multivariable analysis refers to the prediction of a single outcome using multiple variables.

## Types of Data

Traditionally data can be either of the below

### Qualitative vs. Quantitative

- Quantitative Data: It provides numerical information (numbers) or information about quantities; that is, information that can be measured and written down with numbers.
  Examples are: height, size, length, weight etc.

- Qualitative Data: It provides descriptive information (describing something) about qualities; information that can't actually be measured.
  Examples are: Eye Color, Gender etc.

Here are examples covering some differences

- Number of people attended to a party. (Quantitative.)

- The softness of a cat. (Qualitative.)

- The color of the sky. (Qualitative.)

- The number of pennies in your pocket. (Quantitative.)

### Discrete vs. Continuous

> `Quantitative data` can further be classified as `Descrete or Continuous` which are as described below:
>
> Discrete Data which can only take certain values (like whole numbers)
> Continuous Data which can take any value (within a range)
>
> In short, If a variable can take on any value between its minimum value and its maximum value, it is called a continuous variable; otherwise, it is called a discrete variable.


### Univariate vs. Bivariate Data

Statistical data are often classified according to the number of variables being studied.

- Univariate Data:

When we conduct a study that looks at only one variable, we say that we are working with univariate data. Suppose, for example, that we conducted a survey to estimate the average weight of high school students. Since we are only working with one variable (weight), we would be working with univariate data.

- Bivariate Data:

When we conduct a study that examines the relationship between two variables, we are working with bivariate data. Suppose we conducted a study to see if there were a relationship between the height and weight of high school students. Since we are working with two variables (height and weight), we would be working with bivariate data.


## Dependent and Independent Variables

In world of learning, there are two entities.
- Input and
- Output

The output of the learning is based on the input, and hence `output` is coined as a dependent variable and `input` as an independent variable. Consider for example a task of predicting the `Mileage of a Car` based on the parameters like `Make of Car`, `Year of Manufacturing` and `Engine Capacity`.

The input variables are typically denoted using the symbol X, with a subscript to distinguish them. So, 𝑋₁ might be the `Make of Car`, 𝑋₂ might be `Year of manifacturing` and 𝑋₃ might be `Engine Capacity`. The output variables are denoted by the symbol Y.

The input variables may also be known with other names like `predictors`, `independent variables`, `features` or sometimes as just `variables`.
The output variables may also be known as `response` or `dependent` variables.

With the above example of Car, we can write an equation representing some kind of relation between the car mileage (Y) and the input parameters as below.

Y = 𝑤₀𝑋₀ + 𝑤₁𝑋₁ + 𝑤₂𝑋₂ + 𝑤₃𝑋₃
  = bias + 𝑤₁𝑋₁ + 𝑤₂𝑋₂ + 𝑤₃𝑋₃

or Y = f(X) + ϵ       (where ϵ is the error term)

where `bias` is the error term
       the coefficients 𝑤₁, 𝑤₂ and 𝑤₃ are the coefficients of independent variables,
       obtained as the best mathematical fit of specified model or equation.

A coefficient indicates the impact of each independent variable on the outcome variable adjusting for all other independent variables. The model serves two purposes.

  1. It can predict the value of dependent variable for new values of independent variables
  2. It can help describe the relative contribution of each independent variable to dependent variable,
     controlling for the influences of the other independent variable.

## Logistic Regression

`Logistic Regression` also known as `Logit model` or `Logistic model` analyzes the relationship between multiple independent variables and a categorical dependent variable, and estimates the probability of occurrence of an event by fitting data to a Logistic curve.

There are two models within logistic regression as listed below:

  1. Binary Logistic Regression &
  2. Multinomial Logistic Regression

`Binary Logistic Regression` would be typically useful when the dependent variable is dichotomous (can fall into 2 categories like Head & Tail; Pass or Fail etc.) and the independent variables are either Continuous or Categorical.
When the dependent variable is not dichotomous and is comprised of more than 2 categories, a `Multinomial Logistic Regression` may be used.

Despite the name being `Logistic Regression`, it's used in the classification category to predict the discrete output. It's mostly used as a binary class classifier and the binary logistic model is used to estimate the probability of a binary response and it generates the response based on one or more predictors or independent vaiables or features. The algorithm employs a `Logistic function` or `Sigmoid function`.

The `Sigmoid function` is used as a hypothesis function which the machine will use to classify the data and predict labels or the target variables. Here the target variables can be either `0` or `1`.

So, mathematically if `y` is the output label then y ∈ [0, 1]

The hypothesis function can convert the output value to either zero or one and the `sigmoid` exactly does that.

## Sigmoid function

The mathematical equation of `Logistic` or `Sigmoid` function is as shown below

![e1]

In `python` such an equation may be coded and visualized as below.

```python
import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(z):
    a = []
    for x in z:
        a.append(1.0 / (1.0 + math.exp(-x)))
    return a


x = np.arange(-10., 10., 0.2)
s = sigmoid(x)
plt.plot(x, s)
plt.title('Sigmoid function')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks()
plt.yticks()
plt.text(-7.5, 0.85, r'$g(z) = \frac{1}{1 + e^{-z}}$')
plt.grid(True)
plt.show()
```

The above code provides the below output

![alt_text](images/sigmoid.png "sigmoid function")

From the above figure, the below can be inferred

- If the value of `z` is greater than or equal to zero, then the
logistic function gives an output value of one

- If the value of `z` is less than zero, then the logistic function
generates the output zero.

Such inferences can be represented mathematically as follows:

![e2]

This would be the function which may be used for performing the Binary
classification.

Representation of the `Sigmoid` function using `Hypothesis` function:

![e3]

Substitute the value of `z` with ![e4] in the preceding equation,
then the equation ![e1]

is converted to ![e5]

[e1]: https://latex.codecogs.com/gif.latex?g%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-z%7D%7D
[e2]: https://latex.codecogs.com/gif.latex?g%28z%29%20%29%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%20%5Cif%20z%20%5Cgeq%200%20%26%20%5C%5C%200%20%5Cif%20z%20%3C%200%20%26%20%5Cend%7Bmatrix%7D%5Cright.
[e3]:
https://latex.codecogs.com/gif.latex?h_%7B%5Ctheta%7D%28x%29%20%3D%20g%28%5Ctheta%5ET%7Bx%7D%29
[e4]: https://latex.codecogs.com/gif.latex?%5Ctheta%5ET%7Bx%7D
[e5]: https://latex.codecogs.com/gif.latex?h_%7B%5Ctheta%7D%28x%29%20%3D%20g%28%5Ctheta%5ET%7Bx%7D%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-%5Ctheta%7BT%7D%7Dx%7D
