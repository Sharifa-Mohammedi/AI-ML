#Colon Cancer Prediction Model
This repository contains the code to predict the type of colon cancer based on various features like age, family history, biopsy results, etc. 
The project includes data preprocessing, feature engineering, model training, and evaluation steps.

##Steps Followed
1. Data Loading
The project begins by loading two CSV datasets, one containing general patient information and another with colon cancer-related data.
The datasets are merged into a single dataframe to facilitate further analysis and modeling.
2. Exploratory Data Analysis (EDA)
   
Data Exploration:
The initial step in understanding the dataset is data exploration, which provides an overview of the key features and their distributions. This helps identify patterns, relationships between features, and any data quality issues like missing values or outliers. By examining these features, we can gain insights into how the different variables might impact the prediction of colon cancer types.

Visualizations:
Several visualization techniques are used to explore the data in detail:
Histograms: These are used to visualize the distribution of numerical features like AGE and Polyp Size (mm).
Histograms help us understand the spread of values, their skewness, and any potential outliers in the data.

Boxplots: Boxplots are also used for numerical features like AGE and Polyp Size (mm) to visually check for outliers. The boxplots provide a clear picture of the central tendency (median) and variability of the features, as well as any extreme values that fall outside the whiskers, which may be indicative of outliers.
Correlation Heatmaps:
A correlation heatmap is used to visualize the pairwise relationships between numerical features. Correlation measures how strongly two variables are related to each other.
In this analysis, only the numerical columns (such as AGE, Polyp Size (mm), CEA Level, Tumor Grade, Lymph Node Involvement, etc.) are selected to compute the correlation, as these are the features that can be meaningfully correlated.
The heatmap is created using the correlation matrix, which calculates the Pearson correlation coefficient between each pair of numerical columns.
The values range from -1 to 1, where:
+1 indicates a perfect positive correlation,
-1 indicates a perfect negative correlation, and
0 indicates no correlation.
The heatmap allows us to easily identify features that are highly correlated with each other, which could be important for model selection and might suggest potential feature engineering or dimensionality reduction.

Count Plots:
A count plot is used to visualize the distribution of the target variable Type of Colon Cancer. This helps in understanding whether the data is balanced or imbalanced, which is important for choosing the right evaluation metrics and modeling strategies. For example, an imbalanced dataset may require the use of techniques like class weighting or resampling to improve the model's ability to predict the minority class.
Key Observations:
The dataset contains several important features such as AGE, Polyp Size (mm), Family History, Tumor Grade, Lymph Node Involvement, and others. 
These features are essential for predicting the type of colon cancer.
The target variable ("Type of Colon Cancer") shows class imbalance, with one class being more prevalent than the others. 
This class imbalance needs to be addressed during model training using techniques such as resampling or class weighting.
From the correlation heatmap, it was observed that some numerical features, such as Polyp Size (mm) and Tumor Grade, show strong correlations with each other, which suggests that they may carry similar information.
It may be useful to consider these correlations when designing the model or selecting features.
The Age feature appears to have a normal distribution, but Polyp Size may exhibit skewness, which could be important for deciding on appropriate preprocessing steps (e.g., log transformation).

3. Data Preprocessing
Handling Missing Values:
Numerical features with missing values are filled using the median of each respective column.
Categorical features are encoded numerically using label encoding.
Feature Engineering:
Interaction features are created by combining relevant variables. For example, age and polyp size are multiplied to create an interaction term (Age_Polyp_Interaction), which might provide additional predictive value.
Age is categorized into groups: "Young", "Middle-Aged", and "Elderly".
Polyp size is categorized into groups: "Small", "Medium", and "Large", based on typical clinical ranges.
Outlier Handling:
Numerical features such as Age and Polyp Size are capped at the 5th and 95th percentiles to remove extreme values that could skew the model.

4. Model Training
Model Selection:
Two models are selected for training: Random Forest Classifier and Logistic Regression.
Hyperparameter Tuning:
The Random Forest model is tuned using GridSearchCV to find the best combination of hyperparameters, such as the number of estimators and maximum depth.
Model Training:
Both models are trained on the preprocessed data. Logistic Regression is trained with a higher max_iter to ensure convergence.

5. Model Evaluation
The performance of both models is evaluated using several metrics:
Accuracy: The proportion of correct predictions.
Precision: The proportion of true positives among all predicted positives.
Recall: The proportion of true positives among all actual positives.
F1-Score: The harmonic mean of precision and recall, providing a balanced metric.
The Random Forest model performs better than Logistic Regression, so it is selected as the final model.

6. Hyperparameter Tuning (Random Forest)
A hyperparameter grid is defined, and GridSearchCV is used to search for the optimal values for parameters such as n_estimators, max_depth, and min_samples_split.

7. Model Explainability (SHAP)
SHAP (SHapley Additive exPlanations) is used to interpret the Random Forest model’s predictions. SHAP values provide insights into how each feature contributes to individual predictions.
A SHAP summary plot is generated to visually represent the influence of each feature on model predictions.

###Conclusion
This project aims to build a machine learning model to predict the type of colon cancer using patient data. Through various stages, including data exploration, feature engineering, and model training, the project has achieved significant insights into the relationship between features like age, polyp size, and family history.

Key techniques such as hyperparameter tuning were applied to improve model performance. The Logistic Regression model, after fine-tuning, performed fairly well in predicting the type of colon cancer, especially in distinguishing between classes 1 and 2.

Here is a summary of the Logistic Regression performance:

Accuracy: 80%
For class 1 (Type 1 Colon Cancer):
Precision: 0.50
Recall: 0.38
F1-Score: 0.43
For class 2 (Type 2 Colon Cancer):
Precision: 0.86
Recall: 0.94
F1-Score: 0.90
For class 3 (Type 3 Colon Cancer):
Precision: 0.33
Recall: 0.20
F1-Score: 0.25
The Logistic Regression model performed well in classifying Type 2 Colon Cancer (with high precision and recall), but its performance was weaker for Type 1 and Type 3 cancers. This suggests that the model may benefit from further tuning, or more data may be needed for the underrepresented classes to improve its ability to generalize across all types.

Additionally, SHAP (SHapley Additive exPlanations) was utilized to interpret the model, providing valuable insights into how different features contributed to the predictions.<img width="284" alt="Screenshot 2024-11-11 145405" src="https://github.com/user-attachments/assets/8f23b945-f0e6-4d15-b314-0e2e13218752">
<img width="284" alt="Screenshot 2024-11-11 145405" src="https://github.com/user-attachments/assets/8626f945-3df1-46fb-a61a-938cf8d39363">
