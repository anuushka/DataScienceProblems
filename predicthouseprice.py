import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)
y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]
X.head
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))


# Validation data - Split data into train and test
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

train_X, val_X, train_y, val_y = train_test_split(X, y ,random_state = 1)
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)

# print the top few validation predictions
print("First in-sample predictions:", iowa_model.predict(val_X.head()))# print the top few actual prices from validation data
# print the top few validation predictions
print("Actual target values for those homes:", train_y.head().tolist())

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)




