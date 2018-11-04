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

where `z` is the weighted sum defined as under
![e6]

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
plt.text(-7.5, 0.70, r'$z \rightarrow  \infty  ; g(z) \rightarrow 1$')
plt.text(-7.5, 0.65, r'$z \rightarrow  -\infty  ; g(z) \rightarrow 0$')
plt.grid(True)
plt.show()
```

The above code provides the below plot

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

is converted to the following
![e5]


### A pretty useful property of the derivative of Sigmoid function.

Differentiating the `hypothesis` and the `sigmoid`

![e7]

![e8]

which can be re-rewritten as

![e9]

This simplifies to

![e10]

### Parameter estimation

The goal of the Logistic Regression is to find or estimate the unknown
parameters `θ (w₀, w₁, w₂ ... wₙ)` which is done using the __`Maximum Likehood
Estimation`__ which entails finding the set of parameters for which the _Probability_
of the observed data is greatest. The `Maximum Likelihood equation` is derived
from the probability distribution of the dependent variable.

So, in essence, we need to find the coefficient `θ` for the best fit model
which best explains the training data set, by using the `Maximum Likelihood
Estimator` under a set of assumptions. Let's endow our classification model
with a set of probabilistic assumptions and then fit the parameters via
maximum likelihood.

Let us assume the below

![e11]
![e12]

At this point we can discuss about the __Bernoulli Distribution__ regarding
the probability assumptions. `Bernoulli Distribution` is the probability
distribution of a random variable taking on only two values, `1`  (`success`)
and `0` (`false`) with complementary probabilities `p` and `q` respectively.

Where `p` and `q` are related to each other as `p + q = 1` or `q = 1 - p`.

Mathematically,  `Bernoulli Distribution` is the probability distribution of
random variable *X* having rhe `probability density function` defined as below:

- __X__ takes two values `0` and `1`, with probabilities `p` and `1 - p`. That is
![e13]
for `0 < p < 1`

- Frequency function which is the closed form pf probability density function is written as
![e14]

With this, we can express the assumptions made earlier in more compact form as
under:

![e15]

Then the likelihood of the parameter `θ` may be written as:

`L(θ) = P(y|X; θ)`

[e1]: https://latex.codecogs.com/gif.latex?g%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-z%7D%7D
[e2]: https://latex.codecogs.com/gif.latex?g%28z%29%20%29%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%20%5Cif%20z%20%5Cgeq%200%20%26%20%5C%5C%200%20%5Cif%20z%20%3C%200%20%26%20%5Cend%7Bmatrix%7D%5Cright.
[e3]:
https://latex.codecogs.com/gif.latex?h_%7B%5Ctheta%7D%28x%29%20%3D%20g%28%5Ctheta%5ET%7Bx%7D%29
[e4]: https://latex.codecogs.com/gif.latex?%5Ctheta%5ET%7Bx%7D
[e5]: https://latex.codecogs.com/gif.latex?h_%5Ctheta%7B%28x%29%7D%20%3D%20g%28%5Ctheta%5E%7BT%7D%7Bx%7D%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-%5Ctheta%5E%7BT%7D%7Bx%7D%7D%7D
[e6]: https://latex.codecogs.com/gif.latex?z%20%3D%20%5Ctheta%5ET%7Bx%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_0x_0%20&plus;%20w_1x_1%20&plus;%20w_2x_2%20&plus;%20...%20&plus;%20w_nx_n
[e7]: https://latex.codecogs.com/gif.latex?%7Bg%7D%27%28z%29%20%3D%20%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20z%7D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-z%7D%7D
[e8]: https://latex.codecogs.com/gif.latex?%7Bg%7D%27%28z%29%20%3D%20%5Cfrac%7B1%7D%7B%281%20&plus;%20e%5E%7B-z%7D%29%5E2%7D%20%7Be%5E%7B-z%7D%7D
[e9]: https://latex.codecogs.com/gif.latex?%7Bg%7D%27%28z%29%20%3D%20%5Cfrac%7B1%7D%7B%281%20&plus;%20e%5E%7B-z%7D%29%7D%5Cleft%20%28%201%20-%20%5Cfrac%7B1%7D%7B%281%20&plus;%20e%5E%7B-z%7D%29%7D%20%5Cright%20%29
[e10]: https://latex.codecogs.com/gif.latex?%7Bg%7D%27%28z%29%20%3D%20g%28z%29%5Cleft%20%28%201%20-%20g%28z%29%20%5Cright%20%29
[e11]: https://latex.codecogs.com/gif.latex?P%28y%20%3D%201%20%7C%20x%3B%20%5Ctheta%29%20%3D%20h_%7B%5Ctheta%7D%28x%29
[e12]: https://latex.codecogs.com/gif.latex?P%28y%20%3D%200%20%7C%20x%3B%20%5Ctheta%29%20%3D%201%20-%20h_%7B%5Ctheta%7D%28x%29
[e13]: https://latex.codecogs.com/gif.latex?Pr%28X%20%3D%20x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20p%20%26%20x%20%3D%201%20%5C%5C%201%20-%20p%20%26%20x%20%3D%200%20%5Cend%7Bmatrix%7D%5Cright.
[e14]: https://latex.codecogs.com/gif.latex?p%5Exq%5E%7B1%20-%20x%7D%20%3D%20p%5Ex%281%20-%20p%29%5E%7B1%20-%20x%7D
[e15]: https://latex.codecogs.com/gif.latex?P%28y%20%7C%20x%3B%20%7B%5Ctheta%7D%29%20%3D%20%28h_%7B%5Ctheta%7D%28x%29%29%5Ey%281%20-%20%28h_%7B%5Ctheta%7D%28x%29%29%29%5E%7B1%20-%20y%7D
