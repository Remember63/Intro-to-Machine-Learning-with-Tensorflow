#Part 1: Drawing the learning curves
#In here, we'll be comparing three models:

#A Logistic Regression model.
#A Decision Tree model.
#A Support Vector Machine model with an rbf kernel, and a gamma parameter of 1000 (this is another type of model, don't worry about how it works for now).
#Uncomment the code for each one, and examine the learning curve that gets drawn. If you're curious about the code used to draw the learning curves, it's on the utils.py tab.


# Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
#estimator = LogisticRegression()

### Decision Tree
#estimator = GradientBoostingClassifier()

### Support Vector Machine
estimator = SVC(kernel='rbf', gamma=1000)
