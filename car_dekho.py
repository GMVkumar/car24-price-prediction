import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

car_df = pd.read_csv('cardekho_dataset.csv')
car_df.head()


car_df.isnull().sum()

car_df.duplicated().sum()

car_df = car_df.drop(columns=['Unnamed: 0', 'car_name'])
num_cols= car_df.select_dtypes(include=np.number).columns


"""I notice outlier so i apply IQR on outliers"""

def remove_outliers_iqr(car_df, num_cols):
    q1 = car_df[num_cols].quantile(0.25)
    q3 = car_df[num_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return car_df[(car_df[num_cols] >= lower) & (car_df[num_cols] <= upper)]

car_df = remove_outliers_iqr(car_df, 'selling_price')

car_df['selling_price_log'] = np.log1p(car_df['selling_price'])



def cap_outliers(car_df, num_cols):
    lower = car_df[num_cols].quantile(0.01)
    upper = car_df[num_cols].quantile(0.99)
    car_df[num_cols] = car_df[num_cols].clip(lower, upper)
    return car_df

car_df = cap_outliers(car_df, 'selling_price')

car_df.describe()

car_df.drop(columns=['selling_price'], inplace=True)

car_df.rename(columns={'selling_price_log':'selling_price'}, inplace=True)


car_df['brand'].unique()

car_df['model'].unique()

cat_col= car_df.select_dtypes(include='object').columns


le=LabelEncoder()
for col in cat_col:
    car_df[col] = le.fit_transform(car_df[col])



X = car_df.drop(columns=['selling_price'])
y = car_df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

y_pre = model.predict(X_test)


print("R² Score:", r2_score(y_test, y_pre))
print("MAE:", mean_absolute_error(y_test, y_pre))
print("MSE:", mean_squared_error(y_test, y_pre))

"""hyper tuning to get increase model r2score for better prediction"""



param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    cv=2,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print("\nFinal Test R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



joblib.dump(best_xgb, 'xgboost_model.pkl' )