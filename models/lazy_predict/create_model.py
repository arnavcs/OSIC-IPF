from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

# reading the important data from files

train = pd.read_csv('extra/formatted_train.csv')

feature_columns = pd.read_csv('extra/feature_columns.csv').values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]

# format the data to prepare for lazypredict

x = train[feature_columns]
y = train['FVC']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# fit the model

reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)