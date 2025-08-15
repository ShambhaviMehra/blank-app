# Machine Learning based Breast Cancer Prediction (METABRIC)

Breast Cancer is one of the leading causes of death around women. Current diagnostic workflows can be time-consuming and resource intensive, and may not be readily available to certain sections of the population of especially developing countries. The application of Machine Learning technologies into diagnostic systems in hospitals can aid oncologists and other healthcare professionals in predicting key features regarding breast tumours, such as malignancy, tumour stage, chance of recurrence etc. This can greatly streamline the diagnostic process and make efficiency accessible to a larger variety of hospitals and clinics. 

This project aims to predict the tumour stage, probability of recurrence and the survival time of a breast cancer patient using the METABRIC dataset available on Kaggle. The dataset is given [here](https://www.kaggle.com/datasets/gunesevitan/breast-cancer-metabric).

Through this project I used various machine learning models including xgboost regressor, decision trees and logistic regression to predict the different parameters. Each of these gave me a high rate of accuracy, which gave me confidence in the working of my models and prompted me to build my simple ML code into a web application using Streamlit. 

## Methodology

### Libraries Used
For this project, the libraries I used are:
Pandas
Numpy
Scikit-learn
XGBoost

### Data Processing
Initially, I cleaned the data by dropping the null values from the features 'Overall Survival(Months)', 'Overall Survival Status' and 'Tumor Stage', since these were the three features I was trying to prevent as an end goal.Then I mapped the values from Overall Survival Status from string values to numerical values to make it easier for the model to classify them.

### Feature Selection
The features I selected for this project are Age at Diagnosis, Tumor Size, Neoplasm Histologic Grade, Nottingham Prognostic Index and Lymph Nodes examined positive. Given below is the reasoning for each:

1. Age at Diagnosis: Age is a strong predictor of prognosis and treatment outcomes. Younger patients (<40) often have more aggressive cancers, while older patients may have slower-growing tumors. It is a non-invasive, easily available demographic variable that helps stratify patients early.

2. Tumor Size: Larger tumors are generally more advanced and associated with worse prognosis. This metric is also directly linked to likelihood of metastasis and survival rates.

3. Neoplasm Histologic Grade: This measures how abnormal tumor cells look under the microscope (Grade 1 = low, Grade 3 = high)- higher grades mean faster growth and worse prognosis. This is part of standard pathology reports and influences chemotherapy decisions.

4. ER Status (Estrogen Receptor Status): Determines if tumor growth is driven by estrogen. ER-positive cancers respond well to hormone therapy, often with better outcomes. ER-negative cancers tend to be more aggressive.

5. Nottingham Prognostic Index (NPI): Combines tumor size, histologic grade, and lymph node status into a single clinically validated score. It integrates multiple prognostic factors into one number, improving model robustness.

6. Lymph Nodes Examined Positive: Number of lymph nodes with cancer spread is one of the strongest predictors of recurrence and survival. Lymph node positivity indicates possible metastasis.

### Model Training

For training these models, I initially used the train_test_split function from scikit learn to divide the data into the training set and the testing set. Then, I used data pipelines to impute the data wherever it is missing, scale the data and fit it to the respective model. 

For the tumour stage, I used a decision tree classifier, for the Survival time, I used the XGB regressor and for the overall survival status I used Logistic regression. 

## Key Results & Insights

The web application I developed allows users to input values for the metrics I listed above in feature selection and predicts relevant characteristics of the specific patient. 

The current web application prioritizes usability and quick predictions, and therefore does not display performance metrics in the interface. However, during the model training phase, evaluation was performed using standard metrics such as accuracy, precision, recall, and F1-score to ensure reliable predictions. These metrics can be integrated into the app in future iterations to provide end-users with transparency regarding model performance.

## Business Recommendations

This machine learning model could potentially be used by hospitals, diagnostic labs and insurance companies looking to further streamline their process of breast cancer diagnosis. Other uses could potentially be using this on telemedicine platforms and integrating it as a mobile application for general practitioners in places where specialised testing facilities are not readily available and the doctors need a platform to verify the data. However, this should not be used as a primary diagnostic tool but only a supplementary web application to aid with diagnosis. 

## Scalability
This web application can be scaled in a variety of ways such as enabling it to detect multiple types of cancers, integrating it into mobile apps and other telemedicine platforms. This could be a part of the AI in healthcare market which is growing steadily all over the world. 

## Conclusion
I had started this project as a way to practice my elementary machine learning skills and apply them to the healthcare space that I am quite passionate about. This project helped me pick up valuable skills for data analysis and machine learning models and helped me to build something meaningful.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
