
# Regression

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

# Classification

## Consumption Industry

#### KNN on Telecommunications Customer Classification
Create a KNN model for a telecommunication company,to predict group membership using demographic data, predict, for the company can customize offers for individual prospective customers

#### Logistic Regression on Customer Churn
Create a LR model for a telecommunication company, to predict when its customers will leave for a competitor;
measure the accuracy by jaccard and log loss

## Healthcare Industry 

#### Decision Trees on Patients' Drug Classification
build a decision tree model from historical data of patients; 
use the trained model to predict the class of a unknown patient, or to find a proper drug for a new patients

#### SVM on Cancer Classification
Use SVM (Support Vector Machines) to build and train a model using human cell records;
classify cells to whether the samples are benign or malignant; use metrics like f1 score and jaccard 

# Clustering

#### K-Means Clustering on Customer Segementation
classify customers to three segementations using K-Means to understand and customize strategies for retaining those customers

#### Hierarchical Clustering on Car Model
use Agglomerative hierarchical clustering methods, to find the most distinctive clusters of vehicles;
summarize the existing vehicles and help manufacturers to make decision about the supply of new models
