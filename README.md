# Firefly Concerts

![Firefly_Live](https://github.com/user-attachments/assets/d3bcf42c-0ff0-425c-ae7c-431caf1d517e)

## Live Demo
You can access the live application here: [Live at Firefly Beer Predictor](http://live-at-firefly-beer-predictor.s3-website.us-east-2.amazonaws.com/)

## Brief Summary
This project analyzes historical concert data, artist information/demographics with Spotify's API, and historical weather information via Open Meteo API.

The goal initially is to try and predict how much of the six beers would be sold, to aid with inventory management and ordering. Future iterations could span to analytical analysis/viz, cocktail sales, and
any other metric we might be interested in.

## Methodology on Model Selection
1. Data Preparation & Feature Engineering

Raw concert data was processed to reduce noise and emphasize signals that correlate with consumption behavior. Key engineering steps included:

    Logarithmic Transformations: Significant skewness was identified in "Expected Attendance" (GA) and "Artist Followers." A log1p transformation was applied to these features to normalize the distribution and improve model convergence.

    Seasonality & Timing: Concert dates were mapped to seasons (Spring, Summer, Fall, Winter), and days were grouped into binary "Weekend" vs. "Weekday" categories to capture peak consumption periods.

    Feature Pruning: High-cardinality and "leakage" columns—such as raw artist names and non-target beverage sales—were removed to ensure the model generalizes to future shows rather than memorizing historical outliers.

2. The Model Decisions.

To determine the most effective algorithm, a competitive evaluation was performed across four distinct modeling approaches:

    Linear Models (Ridge & Lasso).

    Random Forest Regressor

    Gradient Boosting Regressor

    XGBoost Regressor (Final Selection): Ultimately selected for its superior handling of sparse data and advanced regularization.

3. Evaluation Strategy: LOOCV

Because the dataset represents Firefly's specific concert history, every data point is valuable. We utilized Leave-One-Out Cross-Validation (LOOCV) for the evaluation phase.

In this process, the model is trained N times (where N is the number of concerts), each time leaving out exactly one concert to act as the "unseen" test set. This provided a highly reliable estimate of how the model will perform on the next real-world show.

4. Final Architecture: Multi-Model Regressor

The final production environment utilizes six independent XGBoost models, one specialized for each beer. This "one-model-per-brand" strategy allows the system to account for different consumer behaviors.
5. Hyperparameter Optimization & Safety Buffers

To maximize accuracy, Optuna was employed to conduct a Bayesian search for optimal hyperparameters (learning rate, tree depth, and subsampling). **This will be evaluated next concert season, as the calculated buffers often double inventory rates.**
