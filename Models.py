#%% Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

#%% Set working directory
os.chdir(r'C:\Python Projects\Firefly')




# %% Random Forest Regression Models for Each Beer Type
# Load the data we cleaned
df = pd.read_csv('Beer_Model_DF.csv')

# List of targets to predict
targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']

# Preprocessing: Convert categories (Day, Season, Genre) into numeric dummy variables
# We drop_first=True to avoid the "dummy variable trap" (multi-collinearity)
df_model = pd.get_dummies(df, columns=['Day', 'Season', 'Generic_Genre'], drop_first=True)

results = {}

for beer in targets:
    # 1. Filter: Only train on rows where this specific beer was available
    # This prevents the model from trying to learn from 'NaN' values
    target_df = df_model[df_model[beer].notnull()]
    
    # 2. Define X (Features) and y (Target Beer)
    y = target_df[beer]
    X = target_df.drop(columns=targets)
    
    # 3. Split: 80% Train, 20% Test
    # random_state ensures your results are the same every time you run it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize & Train Model
    # max_depth=5 is used to prevent the model from 'memorizing' your 38 rows
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Predict and Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Save results
    results[beer] = {
        'Model': model,
        'MAE': round(mae, 2),
        'R2_Score': round(r2, 2),
        'Feature_Importance': pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(3).to_dict()
    }

# Display Results Summary
for beer, metrics in results.items():
    print(f"--- {beer} Model ---")
    print(f"Mean Absolute Error: {metrics['MAE']} cans")
    print(f"R-Squared Score: {metrics['R2_Score']}")
    print(f"Top 3 Predictors: {list(metrics['Feature_Importance'].keys())}")
    print("\n")
# %% 

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Hides sklearn warnings for cleaner output

# 1. Load Data
df = pd.read_csv('merged_concert_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. Feature Engineering
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['Season'] = df['Date'].dt.month.map(get_season)

# Group days into Weekend vs Weekday to reduce noise
df['is_weekend'] = df['Weekday_Num'].apply(lambda x: 1 if x >= 5 else 0)

# Availability Flags
df['has_montucky'] = df['Montucky'].notnull().astype(int)
df['has_pacifico'] = df['Pacifico'].notnull().astype(int)

# 3. Log Transforms for Highly Skewed Features (Crucial Step!)
df['log_GA'] = np.log1p(df['GA'])
df['log_SPOTIFY_FOLLOWERS'] = np.log1p(df['SPOTIFY_FOLLOWERS'])

# 4. Drop Leakage and High Cardinality Columns
cols_to_drop = [
    'Vodka_Cocktail', 'Whiskey ', 'ORO', 'Water', 'Soda_Mocktail', 'NA', 
    'RW', 'Agave', 'Wine', 'White Claw', 'PremB/S', 'CLUB', 'VIP ', 
    'DROP', 'ARTIST', 'GENRE', 'Pit_Count', 'DOORS', 'SHOWTIME', 'Date',
    'GA', 'SPOTIFY_FOLLOWERS', 'Day', 'Weekday_Num' # Dropped the raw versions
]
df_clean = df.drop(columns=cols_to_drop)

# 5. Convert Text to Numbers
df_model = pd.get_dummies(df_clean, columns=['Season', 'Generic_Genre'], drop_first=True)
targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']

# 6. Model "Bake-Off" using LOOCV
# We test Linear models alongside Tree models
models_to_test = {
    'Ridge (Linear)': Ridge(alpha=10.0), 
    'Lasso (Linear)': Lasso(alpha=5.0),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
}

results = []
loo = LeaveOneOut()

print("Evaluating Models using Leave-One-Out Cross Validation...\n")

for beer in targets:
    # Filter for when beer was available
    target_df = df_model[df_model[beer].notnull()].copy()
    
    # Skip if almost no data
    if len(target_df) < 10:
        continue
        
    y = target_df[beer].values
    X_raw = target_df.drop(columns=targets)
    
    # Standardize features (Required for Ridge/Lasso to work properly)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    best_model_name = ""
    best_r2 = -float('inf')
    best_mae = float('inf')
    
    for name, model in models_to_test.items():
        # Predict the "left out" row for every single concert
        preds = cross_val_predict(model, X, y, cv=loo)
        
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        
        # Keep track of the champion model
        if r2 > best_r2:
            best_r2 = r2
            best_mae = mae
            best_model_name = name
            
    results.append({
        'Beer': beer,
        'Best Model': best_model_name,
        'MAE': round(best_mae, 2),
        'LOOCV R-Squared': round(best_r2, 2)
    })

# Show final leaderboard
leaderboard = pd.DataFrame(results)
print(leaderboard)

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. LOAD & CLEAN
df = pd.read_csv('merged_concert_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. FEATURE ENGINEERING (The 'Secret Sauce')
# Rain Intensity buckets
def get_rain_intensity(inches):
    if inches == 0: return 'None'
    if inches < 0.2: return 'Light'
    return 'Heavy'

df['Rain_Level'] = df['rain_total_in'].apply(get_rain_intensity)

# Scale log values for Attendance and Spotify
df['log_GA'] = np.log1p(df['GA'])
df['log_Followers'] = np.log1p(df['SPOTIFY_FOLLOWERS'])

# Date-based features
df['is_weekend'] = df['Weekday_Num'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df['Date'].dt.month

# 3. PREPARE MODELING DATAFRAME
cols_to_keep = [
    'Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City', # Targets
    'log_GA', 'log_Followers', 'SPOTIFY_POPULARITY', 'YEARS_ACTIVE',     # Artist
    'High_Temp', 'Show_Duration', 'is_weekend', 'Month',                # External
    'Rain_Level', 'Generic_Genre'                                       # Categorical
]
df_final = pd.get_dummies(df[cols_to_keep], columns=['Rain_Level', 'Generic_Genre'], drop_first=True)

# 4. TRAIN & EVALUATE
targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']
loo = LeaveOneOut()
final_results = []

for beer in targets:
    # Filter for shows where beer was offered
    subset = df_final[df_final[beer].notnull()]
    if len(subset) < 12: continue # Skip if data is too thin
    
    y = subset[beer].values
    X = subset.drop(columns=targets)
    
    # Scaling is crucial for the Gradient Boosting algorithm to weigh features correctly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Hyperparameters tuned for N=38 (Conservative to prevent overfitting)
    model = GradientBoostingRegressor(
        n_estimators=60, 
        learning_rate=0.04, 
        max_depth=2, 
        random_state=42
    )
    
    # Cross-validation
    preds = cross_val_predict(model, X_scaled, y, cv=loo)
    
    # Metrics
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    # Feature Importance to see if 'Rain_Level' helped
    model.fit(X_scaled, y) # Fit on all data to get importance
    top_feature = X.columns[np.argmax(model.feature_importances_)]
    
    final_results.append({'Beer': beer, 'MAE (Error)': round(mae, 1), 'R2': round(r2, 2), 'Main Driver': top_feature})

# SHOW THE IMPROVED LEADERBOARD
print(pd.DataFrame(final_results))




# %% Models that use per capita consumption vs just log of GA

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 1. LOAD & CLEAN
df = pd.read_csv('merged_concert_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. FEATURE ENGINEERING
def get_rain_intensity(inches):
    if inches == 0: return 'None'
    if inches < 0.2: return 'Light'
    return 'Heavy'

df['Rain_Level'] = df['rain_total_in'].apply(get_rain_intensity)

# Scale heavy right-skew features
df['log_GA'] = np.log1p(df['GA'])
df['log_Followers'] = np.log1p(df['SPOTIFY_FOLLOWERS'])

# Date-based features
df['is_weekend'] = df['Weekday_Num'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df['Date'].dt.month

# 3. PREPARE MODELING DATAFRAME
cols_to_keep = [
    'Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City', # Targets
    'GA', # We need raw GA to calculate per capita and scale back
    'log_GA', 'log_Followers', 'SPOTIFY_POPULARITY', 'YEARS_ACTIVE',    # Artist / Attendance
    'High_Temp', 'Show_Duration', 'is_weekend', 'Month',                # External
    'Rain_Level', 'Generic_Genre'                                       # Categorical
]

# One-hot encoding
df_final = pd.get_dummies(df[cols_to_keep], columns=['Rain_Level', 'Generic_Genre'], drop_first=True)

# 4. TRAIN & EVALUATE
targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']
loo = LeaveOneOut()
final_results = []

for beer in targets:
    subset = df_final[df_final[beer].notnull()].copy()
    if len(subset) < 12: 
        continue 
    
    # --- THE ARCHITECTURAL SHIFT: PER CAPITA TARGET ---
    y_true_volume = subset[beer].values
    ga_values = subset['GA'].values
    y_per_capita = y_true_volume / ga_values 
    
    # Drop targets and raw GA from the feature matrix
    X = subset.drop(columns=targets + ['GA'])
    
    # XGBoost configuration (Regularized for small N)
    model = XGBRegressor(
        n_estimators=50, 
        learning_rate=0.05, 
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation predicting PER CAPITA
    preds_per_capita = cross_val_predict(model, X, y_per_capita, cv=loo)
    
    # Convert predictions back to TOTAL VOLUME for real-world evaluation
    preds_volume = preds_per_capita * ga_values
    
    # Evaluate metrics on physical inventory needs
    mae = mean_absolute_error(y_true_volume, preds_volume)
    rmse = root_mean_squared_error(y_true_volume, preds_volume)
    r2 = r2_score(y_true_volume, preds_volume)
    
    # Fit once on full data for feature importance
    model.fit(X, y_per_capita) 
    importance_idx = np.argmax(model.feature_importances_)
    top_feature = X.columns[importance_idx]
    
    final_results.append({
        'Beer': beer, 
        'MAE (Cans)': round(mae, 1), 
        'RMSE (Cans)': round(rmse, 1),
        'R2': round(r2, 2), 
        'Main Driver (Per Capita)': top_feature
    })

# SHOW THE IMPROVED LEADERBOARD
results_df = pd.DataFrame(final_results)
print(results_df.to_markdown(index=False))



# %%
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep console clean

# 1. LOAD & CLEAN
df = pd.read_csv('merged_concert_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. FEATURE ENGINEERING
df['Rain_Level'] = df['rain_total_in'].apply(lambda x: 'None' if x == 0 else ('Light' if x < 0.2 else 'Heavy'))
df['log_GA'] = np.log1p(df['GA'])
df['log_Followers'] = np.log1p(df['SPOTIFY_FOLLOWERS'])
df['is_weekend'] = df['Weekday_Num'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df['Date'].dt.month

cols_to_keep = [
    'Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City',
    'GA', 'log_GA', 'log_Followers', 'SPOTIFY_POPULARITY', 'YEARS_ACTIVE',
    'High_Temp', 'Show_Duration', 'is_weekend', 'Month', 'Rain_Level', 'Generic_Genre'
]
df_final = pd.get_dummies(df[cols_to_keep], columns=['Rain_Level', 'Generic_Genre'], drop_first=True)

targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']
loo = LeaveOneOut()

# 3. OPTUNA OBJECTIVE FUNCTION
def objective(trial, X, y_per_capita, ga_values, y_true_volume):
    # The hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 4),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), # L1 Regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 Regularization
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBRegressor(**param)
    
    # Cross validate predicting per capita
    preds_per_capita = cross_val_predict(model, X, y_per_capita, cv=loo)
    
    # Scale back to volume to calculate the true error
    preds_volume = preds_per_capita * ga_values
    
    # We want Optuna to minimize the MAE of the physical cans
    return mean_absolute_error(y_true_volume, preds_volume)


# 4. RUN THE STUDY FOR EACH BEER
final_results = []
best_models = {}

for beer in targets:
    subset = df_final[df_final[beer].notnull()].copy()
    if len(subset) < 12: 
        continue 
    
    y_true_volume = subset[beer].values
    ga_values = subset['GA'].values
    y_per_capita = y_true_volume / ga_values
    
    X = subset.drop(columns=targets + ['GA'])
    
    # Create an Optuna study for this specific beer
    print(f"--- Optimizing {beer} ---")
    study = optuna.create_study(direction='minimize')
    
    # Wrap the objective to pass our specific data
    study.optimize(lambda trial: objective(trial, X, y_per_capita, ga_values, y_true_volume), n_trials=50)
    
    # Extract the absolute best hyperparameters found
    best_params = study.best_params
    print(f"Best Params for {beer}: {best_params}\n")
    
    # Re-train with the best parameters to get our final metrics
    best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    preds_per_capita = cross_val_predict(best_model, X, y_per_capita, cv=loo)
    preds_volume = preds_per_capita * ga_values
    
    mae = mean_absolute_error(y_true_volume, preds_volume)
    rmse = root_mean_squared_error(y_true_volume, preds_volume)
    
    # --- DYNAMIC SAFETY BUFFER CALCULATION ---
    # Find where the model UNDER-predicted (y_true > y_pred)
    errors = y_true_volume - preds_volume
    under_preds = errors[errors > 0]
    
    if len(under_preds) > 0:
        # Calculate the 90th percentile of our misses to use as a buffer
        buffer_cans = np.percentile(under_preds, 90)
        avg_pred = preds_volume.mean()
        buffer_pct = (buffer_cans / avg_pred) * 100
    else:
        buffer_pct = 0.0
        
    final_results.append({
        'Beer': beer, 
        'Tuned MAE': round(mae, 1), 
        'Tuned RMSE': round(rmse, 1),
        'Suggested Buffer %': f"+{round(buffer_pct, 1)}%"
    })

# SHOW THE IMPROVED LEADERBOARD
results_df = pd.DataFrame(final_results)
print(results_df.to_markdown(index=False))

#%% 
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# 1. LOAD & CLEAN
df = pd.read_csv('merged_concert_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

df['Rain_Level'] = df['rain_total_in'].apply(lambda x: 'None' if x == 0 else ('Light' if x < 0.2 else 'Heavy'))
df['log_GA'] = np.log1p(df['GA'])
df['log_Followers'] = np.log1p(df['SPOTIFY_FOLLOWERS'])
df['is_weekend'] = df['Weekday_Num'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df['Date'].dt.month

cols_to_keep = [
    'Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City',
    'GA', 'log_GA', 'log_Followers', 'SPOTIFY_POPULARITY', 'YEARS_ACTIVE',
    'High_Temp', 'Show_Duration', 'is_weekend', 'Month', 'Rain_Level', 'Generic_Genre'
]
df_final = pd.get_dummies(df[cols_to_keep], columns=['Rain_Level', 'Generic_Genre'], drop_first=True)

targets = ['Modelo', 'Pacifico', 'Corona', 'Seltzer', 'Montucky', 'Holy_City']
loo = LeaveOneOut()

# PLUG IN YOUR BEST PARAMS HERE FROM YOUR OPTUNA RUN
optuna_best_params = {
    'Modelo': {'n_estimators': 31, 'learning_rate': 0.07626971518512268, 'max_depth': 4, 'subsample': 0.7427330886227871}, 
    'Pacifico': {'n_estimators': 119, 'learning_rate': 0.026953648839011655, 'max_depth': 3, 'subsample': 0.9982432001434071}, 
    'Corona': {'n_estimators': 35, 'learning_rate': 0.02542091750802898, 'max_depth': 3, 'subsample': 0.6031108321191261},
    'Seltzer': {'n_estimators': 114, 'learning_rate': 0.06876055314047423, 'max_depth': 1, 'subsample': 0.5849379123236327},
    'Montucky': {'n_estimators': 109, 'learning_rate': 0.012319948667548068, 'max_depth': 4, 'subsample': 0.6916008003860615},
    'Holy_City': {'n_estimators': 81, 'learning_rate': 0.03188724886674788, 'max_depth': 31, 'subsample': 0.54347459326399}
}

metadata = {}

for beer in targets:
    print(f"Exporting {beer}...")
    subset = df_final[df_final[beer].notnull()].copy()
    
    y_true_volume = subset[beer].values
    ga_values = subset['GA'].values
    y_per_capita = y_true_volume / ga_values
    
    X = subset.drop(columns=targets + ['GA'])
    feature_names = list(X.columns)
    
    # Instantiate the optimized model
    model = XGBRegressor(**optuna_best_params.get(beer, {}), random_state=42, n_jobs=-1)
    
    # 1. Calculate Out-of-Sample Buffer using Cross-Validation
    preds_per_capita = cross_val_predict(model, X, y_per_capita, cv=loo)
    preds_volume = preds_per_capita * ga_values
    
    errors = y_true_volume - preds_volume
    under_preds = errors[errors > 0]
    buffer_pct = 0.0
    if len(under_preds) > 0:
        buffer_cans = np.percentile(under_preds, 90)
        buffer_pct = (buffer_cans / preds_volume.mean())
    
    # 2. FIT ON 100% OF THE DATA FOR PRODUCTION
    model.fit(X, y_per_capita)
    
    # 3. SAVE THE MODEL AS .PKL
    # joblib is much more efficient than pickle for tree-based arrays
    joblib.dump(model, f'model_{beer.lower()}.pkl')
    
    # 4. STORE METADATA
    metadata[beer] = {
        'buffer_multiplier': round(buffer_pct, 4), # e.g., 0.9310 for 93.1%
        'features': feature_names # Critical for AWS Lambda
    }

# Save metadata as JSON
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Successfully exported 6 .pkl files and model_metadata.json")
# %%
