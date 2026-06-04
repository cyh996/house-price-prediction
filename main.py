import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error
import numpy as np
import optuna
import os
import joblib

os.makedirs("models", exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv(
    'data/train.csv',
    encoding='latin1'
)

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['GarageType'] = df['GarageType'].fillna('None')
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['GarageQual'] = df['GarageQual'].fillna('None')
df['GarageCond'] = df['GarageCond'].fillna('None')
df['PoolQC'] = df['PoolQC'].fillna('None')
df['Fence'] = df['Fence'].fillna('None')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['BsmtQual'] = df['BsmtQual'].fillna('None')
df['BsmtCond'] = df['BsmtCond'].fillna('None')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')
df['Alley'] = df['Alley'].fillna('None')
mas_mode = df["MasVnrType"].mode()[0]

df.loc[(df["MasVnrType"].isnull()) & (df["MasVnrArea"] > 0), "MasVnrType"] = mas_mode

df["MasVnrType"] = df["MasVnrType"].fillna("None")

bsmt_mode = df["BsmtFinType2"].mode()[0]

df.loc[(df["BsmtFinType2"].isnull()) & (df["BsmtFinSF2"] > 0), "BsmtFinType2"] = bsmt_mode

df["BsmtFinType2"] = df["BsmtFinType2"].fillna("None")

df.loc[(df['BsmtExposure'].isnull()) & (df['TotalBsmtSF'] > 0), 'BsmtExposure'] = 'No'
df['BsmtExposure'] = df['BsmtExposure'].fillna('None')


features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'Neighborhood', 
                    'KitchenQual', 'BsmtQual', 'FireplaceQu', 'BsmtExposure', 'MSZoning']

result = []

def objective(trial):
    model = XGBRegressor(
        n_estimators = trial.suggest_int('n_estimators', 100, 500),
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth = trial.suggest_int('max_depth', 3, 8),
        subsample = trial.suggest_float('subsample', 0.7, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0),
        random_state = 42,
        eval_metric='rmse',
        verbosity=0
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    score = cross_val_score(
       pipe,
       X_train,
       y_train,
       cv = 5,
       scoring='r2'
    ).mean()

    return score

X = df[features]
y = np.log1p(df['SalePrice'])

cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(direction='maximize', sampler = sampler)
study.optimize(objective, n_trials = 50)

print(study.best_params)
print(study.best_value)

best_params = study.best_params

best_model = XGBRegressor(
    **best_params,
    random_state = 42,
    eval_metric = 'rmse',
    verbosity=0
)

best_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, "models/xgb_house_price_model.pkl")

pred = best_pipe.predict(X_test)

pred_price = np.expm1(pred)
y_test_price = np.expm1(y_test)

pred_price = np.maximum(pred_price, 0)

result.append({
    'model': 'XGBRegressor',
    'MAE': mean_absolute_error(y_test_price, pred_price),
    'RMSE': root_mean_squared_error(y_test_price, pred_price),
    'RMSLE': root_mean_squared_log_error(y_test_price, pred_price),
    'R2_Score': r2_score(y_test_price, pred_price)
})

result_df = pd.DataFrame(result)
print(result_df)