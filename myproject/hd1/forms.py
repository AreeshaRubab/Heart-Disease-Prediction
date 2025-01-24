# forms.py
from django import forms

class UserInputForm(forms.Form):
    BMI = forms.FloatField(label='BMI')
    SleepTime = forms.FloatField(label='SleepTime')
    