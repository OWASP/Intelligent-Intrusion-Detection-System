from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
#loading our trained model
print("Keras model loading.......")
model = load_model('attack_labe.hdf5') #after training #TODO
print("Model loaded!!")

    def request():
        return HttpResponse("<h1>Intelligent Intrusion Detection System</h1>")
         return HttpResponse("The predicted attacked label is {}".format(label)) #label will be imported after finishing dvevelopment of analyzer
         
         