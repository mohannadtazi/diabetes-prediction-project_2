from django.shortcuts import render
from django.http import HttpResponse
import pickle
import pandas as pd
def index(request):

 return render(request, "prediction/index.html")


def getprediction(pregnancy, glucose, bloodp, skint, insulin, bmi, diabetes, age):
    # Load the model and the columns
    model = pickle.load(open('ml_model.sav', 'rb'))
    columns = pickle.load(open('columns.sav', 'rb'))
    # Create a dataframe from the data
    data = [[pregnancy, glucose, bloodp, skint, insulin, bmi, diabetes, age]]
    df = pd.DataFrame(data, columns=columns)
    # Make a prediction
    prediction = model.predict(df)
    if prediction[0] == 0:
        return "You are not Diabetic"
    elif prediction[0] == 1:
        return "You are Diabetic"
    else:
        return "Error in prediction"
     

def result(request):
 pregnancy= float(request.POST.get('pregnancy',1))
 glucose= float(request.POST.get('glucose',1))
 bloodp= float(request.POST.get('bloodp',1))
 skint= float(request.POST.get('skint',1))
 insulin= float(request.POST.get('insulin',1))
 bmi= float(request.POST.get('bmi',1))
 diabetes= float(request.POST.get('diabetes',1))
 age= int(request.POST.get('age',1))
 resultat = getprediction(pregnancy, glucose, bloodp, skint, insulin, bmi, diabetes, age)
 return render(request, "prediction/result.html" , context={"result": resultat})

