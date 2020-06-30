
## I Regression

#### Simple & Multiple & Polynomial Regressions on Fuel Consumption
Implement simple, multiple, and polynomial linear regressions to fuel consumption and Carbon dioxide emission of cars;
split fuel consumption data into training and test sets, create models using training set;
evaluate the models using test set by metrics including Mean absolute error (MAE), Residual sum of squares (MSE) and R2-score; and finally to predict unknown value
```
viz = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.head(10)
```
```
plt.scatter(viz.ENGINESIZE,viz.CO2EMISSIONS, color = 'skyblue')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
```
##### Creating train and test dataset
```
mask = np.random.rand(len(df)) < 0.8
train = viz[mask]
test = viz[~mask]
```
```
'skyblue')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
```
#### Modeling
```

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x,train_y )

# The coefficients
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)
```
#### Plot outputs
```
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'skyblue')
plt.plot(train_x, regr.coef_[0][0]*train_x+ regr.intercept_[0], '-b')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
```
#### Evaluation
* Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
* Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
* Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
* R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
```
test_x = np.asanyarray(train[['ENGINESIZE']])
test_y = np.asanyarray(train[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual Sum of Squares (MSE): %.2f" % np.mean((test_y_hat - test_y) **2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y) )
```
#### 2 Multiple Regression Model
```
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y )

# The coefficients
print("Coefficients: ", regr.coef_)
```
#### Prediction
```
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
print("Residual Sum of Squares: %.2f" % np.mean((y_hat - y)**2))
# Explained variance score: 1 is perfect prediction
print("Variance Score: %.2f" % regr.score(x,y))
```
#### 3 Polynomial Regression Mode
```
from sklearn.preprocessing import PolynomialFeatures

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly
```
```
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly,train_y)
# The coefficient
print("Coefficient: ", clf.coef_)
print("Intercept: ", clf.intercept_)
```
```
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'skyblue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)

plt.plot(XX, yy, '-b')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
```
#### Evaluation
```
from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_hat = clf.predict(test_x_poly)

print("Mean absolute error (MAE): %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
```
```
oly3 = PolynomialFeatures(degree = 3)
train_x_poly3 = poly3.fit_transform(train_x)
train_x_poly3

clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3,train_y)
# The coefficient
print("Coefficients: ", clf3.coef_)
print("Intercept: ", clf3.intercept_)
```
```
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)

plt.plot(XX, yy, '-b')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
```
#### Evaluation
```
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_hat = clf3.predict(test_x_poly3)

print("Mean absolute error (MAE): %.2f" % np.mean(np.absolute(test_y3_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_hat , test_y) )
```
## II Classification

### Consumption Industry

#### KNN on Telecommunications Customer Classification
Create a KNN model for a telecommunication company,to predict group membership using demographic data, predict, for the company can customize offers for individual prospective customers

#### Logistic Regression on Customer Churn
Create a LR model for a telecommunication company, to predict when its customers will leave for a competitor;
measure the accuracy by jaccard and log loss

### Healthcare Industry 

#### Decision Trees on Patients' Drug Classification
build a decision tree model from historical data of patients; 
use the trained model to predict the class of a unknown patient, or to find a proper drug for a new patients

#### SVM on Cancer Classification
Use SVM (Support Vector Machines) to build and train a model using human cell records;
classify cells to whether the samples are benign or malignant; use metrics like f1 score and jaccard 

## III Clustering

#### K-Means Clustering on Customer Segementation
classify customers to three segementations using K-Means to understand and customize strategies for retaining those customers

#### Hierarchical Clustering on Car Model
use Agglomerative hierarchical clustering methods, to find the most distinctive clusters of vehicles;
summarize the existing vehicles and help manufacturers to make decision about the supply of new models
