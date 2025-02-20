import numpy as np
import pandas as pd
from sklearn import linear_model
import math


# def gradient_descent(x, y):
#     curr_m = 0
#     curr_b = 0
#     n = len(x)
#     iterations = 1000
#     learning_rate = 0.08
#     for i in range(iterations):
#         y_predicted = curr_m*x + curr_b
#         cost = (1/n) * sum([val ** 2 for val in (y-y_predicted)])
#         m_deriv = -(2/n) * sum(x*(y-y_predicted))
#         b_deriv = -(2/n) * sum((y-y_predicted))
#         curr_m = curr_m - learning_rate * m_deriv
#         curr_b = curr_b - learning_rate * b_deriv
#         print(f"m = {curr_m}, b = {curr_b}, iterations = {i}, COST = {cost}")

# x = np.array([1,2,3,4,5])
# y = np.array([5,7,9,11,13])

# gradient_descent(x,y)

df = pd.read_csv("test_scores.csv")

math_score = df["math"].to_numpy()
cs_score = df["cs"].to_numpy()

def gradient_descent(x,y):
    curr_m = 0
    curr_b = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    prev_cost = 0
    for i in range(iterations):
        y_predicted = curr_m*x+curr_b
        cost = (1/n) * sum([val**2 for val in y-y_predicted])
        m_deriv = -(2/n) * sum(x*(y-y_predicted))
        b_deriv = -(2/n) * sum(y-y_predicted)
        curr_m = curr_m - learning_rate * m_deriv
        curr_b = curr_b - learning_rate * b_deriv
        if math.isclose(cost,prev_cost, rel_tol=1e-20):
            break
        prev_cost = cost
    
    return f"m = {curr_m}, b = {curr_b}, iterations = {i}, COST = {cost}"

print(gradient_descent(math_score,cs_score))

reg = linear_model.LinearRegression()
reg.fit(X=df[["math"]],y=df["cs"])

print(f"Using sklearn:- b = {reg.intercept_}, m = {reg.coef_[0]}")