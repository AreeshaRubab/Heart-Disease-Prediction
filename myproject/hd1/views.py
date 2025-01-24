# views.py
from django.shortcuts import render
import numpy as np
from django.http import HttpResponse
from .forms import UserInputForm
from .models import Preprocessing, FeatureSelection, ConfusionMatrix


def home_screen_view(request):
    result = None
    confusion_matrix_result = None

    if request.method == 'POST':
        form = UserInputForm(request.POST)

        if form.is_valid():
            BMI = form.cleaned_data['BMI']
            SleepTime = form.cleaned_data['SleepTime']
            
            df = Preprocessing.preprocess_data()
            
            selected_features = FeatureSelection.decision_tree_feature_selection(df)

            X_input = np.array([[BMI, SleepTime]])
            prediction = ConfusionMatrix.logistic_regression_predict(df, X_input, selected_features)

            str = ""
            if prediction == 0:
                str = "Congratulations! You donot have Heart Disease"
            elif prediction == 1:
                str = "Ops! You have Heart Disease"

            result = f"Prediction: {str}"
    else:
        form = UserInputForm()

    return render(request, 'base.html', {'form': form, 'result': result, 'confusion_matrix': confusion_matrix_result})


