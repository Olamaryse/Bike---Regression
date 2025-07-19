# Seoul Bike Sharing Demand Prediction

## Project Overview

This project focuses on predicting the demand for Seoul's public bicycle rental service, "Seoul Bike," based on various environmental and temporal factors. The goal is to develop a robust regression model that can accurately forecast bike rental counts, which can assist in optimizing bike distribution and resource allocation.

## Dataset

The dataset used for this project is the "Seoul Bike Sharing Demand" dataset, available on the UCI Machine Learning Repository. It contains hourly bike rental counts along with corresponding weather conditions and seasonal information.

**Key Features:**

  * **Rented Bike Count:** The target variable, representing the number of bikes rented per hour.
  * **Hour:** The hour of the day.
  * **Temperature (C):** Temperature in Celsius.
  * **Humidity (%):** Percentage of humidity.
  * **Windspeed (m/s):** Wind speed in meters per second.
  * **Visibility (10m):** Visibility in 10-meter increments.
  * **Dew Point Temperature (C):** Dew point temperature in Celsius.
  * **Solar Radiation (MJ/m2):** Solar radiation.
  * **Rainfall (mm):** Rainfall amount.
  * **Snowfall (cm):** Snowfall amount.
  * **Seasons:** Categorical variable (Winter, Spring, Summer, Autumn).
  * **Holiday:** Binary variable (Holiday/No holiday).
  * **Functional Day:** Binary variable (Functional hours/Non-functional hours).

## Technologies Used

  * **Python:** The primary programming language for data analysis and model development.
  * **Pandas:** For data manipulation and analysis.
  * **NumPy:** For numerical operations.
  * **Matplotlib & Seaborn:** For data visualization.
  * **Scikit-learn:** For machine learning models (e.g., Linear Regression, StandardScaler).
  * **Imblearn (RandomOverSampler):** Potentially used for handling imbalanced data, though not explicitly shown in the initial snippet for regression, it indicates awareness of data balancing techniques.
  * **TensorFlow:** For building and training neural network models.

## Data Preprocessing and Exploration

1.  **Loading and Initial Inspection:** The dataset was loaded, and irrelevant columns like 'Date', 'Holiday', and 'Seasons' were dropped for this specific analysis.

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    # from imblearn.over_sampling import RandomOverSampler # Uncomment if used for other tasks
    # import tensorflow as tf # Uncomment if used for other tasks

    dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
    df = pd.read_csv("sample_data/SeoulBikeData(in).csv").drop(["Date", "Holiday", "Seasons"], axis=1)
    df.columns = dataset_cols
    df["functional"] = (df["functional"] == 'Yes').astype(int)
    print(df.head())
    ```

    The `functional` column, which was originally 'Yes'/'No', was converted to a numerical representation (1/0).

2.  **Hourly Data Filtering:** The analysis was narrowed down to focus specifically on bike rental patterns at 12:00 PM (noon). This step simplifies the problem by analyzing a specific peak hour.

    ```python
    df = df[df["hour"] == 12]
    df = df.drop("hour", axis=1)
    print(df.head())
    ```

3.  **Feature Visualization:** Scatter plots were generated to visualize the relationship between each independent variable and the 'bike\_count'. This step is crucial for understanding feature distributions and their potential correlation with the target variable.

    *Example Plot (Temperature vs. Bike Count):*

    ```python
    for label in df.columns[1:]:
      plt.scatter(df[label], df["bike_count"])
      plt.title(label)
      plt.ylabel("bike_count")
      plt.xlabel(label)
      plt.show()
    ```

    (Note: The actual plots are generated in the Jupyter Notebook and would be displayed here in a live environment or as static images in a full report.)

## Model Development (Regression)

The notebook demonstrates the application of regression techniques to predict bike demand. While the full model training and evaluation are not explicitly detailed in the provided snippets, the presence of `sklearn.linear_model.LinearRegression` and `tensorflow` imports suggests a comprehensive approach to model building, potentially including:

  * **Data Splitting:** Dividing the dataset into training and testing sets.
  * **Feature Scaling:** Using `StandardScaler` to normalize numerical features, which is essential for many machine learning algorithms.
  * **Model Training:** Training a `LinearRegression` model. The `all_reg.score(x_train_all, y_train_temp)` line indicates model evaluation on the training data.
  * **Evaluation:** Assessing model performance using appropriate regression metrics (e.g., R-squared, Mean Absolute Error, Mean Squared Error).

## Key Learnings and Future Work

This project demonstrates proficiency in data loading, preprocessing, exploratory data analysis (EDA) through visualization, and the application of regression models for predictive tasks.

**Potential Future Enhancements:**

  * **Advanced Feature Engineering:** Explore creating more complex features from existing ones (e.g., interaction terms, polynomial features).
  * **Time Series Analysis:** Given the hourly nature of the data, applying time series models (e.g., ARIMA, Prophet) could yield more accurate predictions.
  * **Deep Learning Models:** Further leverage TensorFlow to build and experiment with more complex neural network architectures for improved predictive power.
  * **Hyperparameter Tuning:** Optimize model performance through systematic hyperparameter tuning.
  * **Deployment:** Explore deploying the trained model as a web service for real-time predictions.
