import os
import django
from django.db import models
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()

class Preprocessing(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    method_used = models.CharField(max_length=255)

    @staticmethod
    def preprocess_data():
        df = pd.read_csv('C:\heart_2020_1.csv')
        df.dropna(inplace=True)
        label_encoder = LabelEncoder()
        columns_to_encode = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

        for column in columns_to_encode:
            df[column] = label_encoder.fit_transform(df[column])

        scaler = StandardScaler()
        numerical_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'AgeCategory', 'Race', 'SleepTime']
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        return df

class FeatureSelection(models.Model):
    preprocessing = models.OneToOneField(Preprocessing, on_delete=models.CASCADE)
    method_used = models.CharField(max_length=255)
    selected_features = models.TextField()

    @staticmethod
    def decision_tree_feature_selection(df):
        X = df.drop(columns=['HeartDisease'])
        y = df['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        sfm = SelectFromModel(clf, threshold=0.1)
        sfm.fit(X_train, y_train)

        selected_features = list(X.columns[sfm.get_support()])

        return selected_features

class ConfusionMatrix(models.Model):
    feature_selection = models.OneToOneField(FeatureSelection, on_delete=models.CASCADE)
    true_positive = models.IntegerField()
    true_negative = models.IntegerField()
    false_positive = models.IntegerField()
    false_negative = models.IntegerField()

    @staticmethod
    def generate_confusion_matrix(df, selected_features):
        X = df[selected_features]
        y = df['HeartDisease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)
        return conf_matrix

    @staticmethod
    def logistic_regression_predict(df, X_input, selected_features):
        clf = LogisticRegression()
        X = df[selected_features]
        y = df['HeartDisease']

        clf.fit(X, y)

        prediction = clf.predict(X_input)
        return prediction
