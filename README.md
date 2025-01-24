# Heart-Disease-Prediction
## Overview  
This project is designed to predict the likelihood of heart disease using machine learning models and a user-friendly Django-based web interface. By processing health-related data, the application assesses the presence or absence of heart disease based on input factors like BMI and sleep time.  

## Dataset  
The project utilizes the **heart_2020_1.csv** dataset, which includes various health and lifestyle factors. With 18 variables, the dataset is leveraged to predict heart disease likelihood. Key features include BMI, PhysicalHealth, MentalHealth, SleepTime, and more.  

## Data Preprocessing  
**Steps:**  
1. **Encoding:** Categorical columns are transformed using `LabelEncoder`.  
2. **Standardization:** Numerical features like BMI, PhysicalHealth, and SleepTime are standardized using `StandardScaler`.  
3. **Cleaning:** Missing values are removed to maintain data quality for model training.  

## Feature Selection  
A Decision Tree Classifier identifies the most relevant features for prediction. Features that exceed a predefined importance threshold are selected for model training. The feature importance is visualized using bar charts to highlight the selected attributes.  

## Classification  
- **Model Used:** Logistic Regression Classifier  
- The model is trained on selected features and tested for performance. Predictions are based on input details like BMI and sleep time, estimating the likelihood of heart disease.  

## Performance Metrics  
The model's accuracy is evaluated using a **Confusion Matrix**, which analyzes true positives, true negatives, false positives, and false negatives.  

## Web Interface  
The project features an intuitive web interface built using Django.  

**Technologies Used:**  
- **Backend:** Python, Django Framework  
- **Frontend:** HTML, CSS with responsive design  

**Key Files:**  
- **`settings.py`:** Contains project configuration.  
- **`models.py`:** Defines data models for preprocessing, feature selection, and evaluation.  
- **`forms.py`:** Captures user inputs for predictions.  
- **`views.py`:** Processes inputs, runs the prediction model, and renders results.  
- **`base.html`:** Provides the main user interface.  
- **`urls.py`:** Handles URL routing within the app.  

## Predicate Logic  
The project incorporates rule-based predicate logic for analysis:  
- **Rule 1:** High BMI and low sleep time increase the likelihood of heart disease.  
- **Rule 2:** Normal BMI and sufficient sleep time decrease the likelihood of heart disease.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Faiza-syed/Heart-Disease-Detection.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd Heart-Disease-Detection  
   ```  
3. Install the required Python packages:  
   ```bash  
   pip install -r requirements.txt  
   ```  
4. Start the Django development server:  
   ```bash  
   python manage.py runserver  
   ```  
5. Access the web interface at: `http://127.0.0.1:8000/`.  

## Usage  
1. Upload health data in the required format.  
2. Enter details like BMI and sleep time into the provided form.  
3. View prediction results and relevant metrics.  

## Acknowledgments  
- **Libraries Used:** scikit-learn, NumPy, pandas, Django  
- **Inspiration:** Health-focused machine learning applications  

For additional details, refer to the project documentation and source code.
