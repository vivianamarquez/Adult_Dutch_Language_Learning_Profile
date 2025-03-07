# 🗣️ Predicting Dutch Speaking Proficiency in Adult Language Learners

## 📌 Project Overview

This project aims to **predict the speaking proficiency of adult learners** taking the **State Examination of Dutch as a Second Language (STEX)** using **machine learning models**. By analyzing linguistic, and educational data, we identify the strongest predictors of language proficiency and provide insights that can guide **educators, policymakers, and language learners** in optimizing Dutch language acquisition.

The dataset, originally compiled by **Schepens, van Hout, and Jaeger (2020)** and available on [Kaggle](https://www.kaggle.com/datasets/thedevastator/adult-language-learning-profile/data?select=stex.csv) [(Kaggle notebook)](https://www.kaggle.com/code/vivianamarquez/predicting-dutch-speaking-proficiency-using-ml/notebook), contains **anonymized exam results** and **demographic information** of learners residing in the Netherlands.

## 📖 Notebooks

Explore our Jupyter notebooks with in-depth analysis and insights:

- [**Model and insights**](https://github.com/vivianamarquez/Adult_Dutch_Language_Learning_Profile/blob/main/Predicting_Dutch_Speaking_Proficiency.ipynb) – Data analysis, feature engineering, and model building.
- [**Ideas for implementation**](https://github.com/vivianamarquez/Adult_Dutch_Language_Learning_Profile/blob/main/Implementation.ipynb) – How this research can be applied in real-world settings.
- [**En español: modelo y conclusiones**](https://github.com/vivianamarquez/Adult_Dutch_Language_Learning_Profile/blob/main/HolandesProyecto.ipynb) – Spanish version with key findings and model details.


## 🔑 Key Features Considered

- **Linguistic background** (e.g., native language, multilingualism)
- **Demographic factors** (e.g., age of arrival, length of residence)
- **Education & Enrollment** (e.g., formal schooling, participation in Dutch courses)

## 🚀 Project Workflow

The project follows a structured **machine learning pipeline**:

1. **Data Loading** – Load and preprocess the dataset.
2. **Exploratory Data Analysis** – Identify trends, correlations, and distributions.
3. **Train/Test Split** – Partition the dataset for model evaluation.
4. **Feature Engineering** – Transform categorical and numerical features.
5. **Regression Modeling** – Train multiple regression models.
6. **Hyperparameter Tuning** – Optimize models for better performance.
7. **Model Selection** – Choose the best-performing model.
8. **Model Interpretation** – Use SHAP values to explain feature importance.
9. **Results & Analysis** – Evaluate model accuracy and key findings.
10. **Model Deployment** – Save and export the trained model.

## 📊 Key Findings

- **Enrollment in Dutch language courses is the strongest predictor** of speaking proficiency. Structured learning is significantly more effective than passive exposure.
- **Native speakers of Germanic languages** (e.g., German, English, Swedish) perform best, while **non-Germanic Indo-European and non-Indo-European speakers** face greater challenges.
- **Monolingual individuals struggle more**, whereas those with a **second language, especially a Germanic one, have an advantage**.
- **Age of arrival matters** – Younger arrivals tend to perform slightly better in Dutch proficiency tests.
- **Length of residence alone has a weak effect** – Long-term residence only improves proficiency when combined with **active learning**.
- **General formal education has minimal impact** – Learning Dutch through structured courses is more influential than overall schooling background.

## 🎯 Business & Policy Recommendations

To improve Dutch language proficiency among learners, **policymakers and educators** should consider:

✅ **Increasing accessibility and funding for Dutch language courses** – Since formal enrollment is the strongest predictor of proficiency.  
✅ **Offering tailored curricula for high-risk language groups** – Non-Indo-European speakers (e.g., Arabic, Chinese, Turkish) face significant challenges.  
✅ **Promoting early exposure programs** – Younger arrivals have better outcomes; early intervention can boost long-term success.  
✅ **Encouraging multilingual education** – Bilingual individuals perform better, suggesting benefits to fostering multilingual skills.  
✅ **Focusing on active language learning over general schooling** – Dutch language training is far more effective than formal education alone.  

## 🔬 Future Work

- **Expand data collection** to include more diverse linguistic backgrounds.
- **Investigate motivational and socioeconomic factors** influencing proficiency.
- **Conduct a longitudinal study** to track language acquisition before and after course enrollment.
- **Enhance predictive models using AI** for personalized language learning plans.

## 🛠️ Tech Stack

- **Python**: pandas, NumPy, scikit-learn, SHAP
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning Techniques**: Linear Regression, Lasso Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, Neural Networks
- **Model Evaluation Metrics**: RMSE, R² Score, Cross-Validation
- **Feature Engineering**: One-Hot Encoding, Scaling, Dimensionality Reduction
- **Web Application & Deployment**: Flask, Jinja2, OpenAI API, Pickle


#### 👥 Project Contributors

This project was developed by:
- [Yohan Jair Rivera Farías](https://www.linkedin.com/in/yohan-rivera-12b847261/)
- Fabián Hernando Rivera Farías 

Computer Science students at Unicomfacauca in Popayán, Colombia 🇨🇴 

It was completed as part of their final project for the AI4All elective, under the supervision of Professor [Viviana Márquez](https://www.linkedin.com/in/vivianamarquez/). 
