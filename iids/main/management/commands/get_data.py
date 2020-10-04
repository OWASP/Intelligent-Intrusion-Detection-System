from django.core.management.base import BaseCommand, CommandError
from preprocessor.preprocessors import *
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import torch
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
from preprocessor.preprocessors import *
from classifier.mlclassifiers import *
from classifier.nnclassifiers import *
import shutil
import os
#import all from others

class Command(BaseCommand):
    
    help = 'provide the path of dataset and config.json '

    def add_arguments(self, parser):
        parser.add_argument('-d', '--dataset', type=dir , help='Provide the dataset for training')
        parser.add_argument('-c', '--config', type=dir, help='Provide the config file of Model')
        parser.add_argument('-m', '--model_type ', type=str, help='Mention the type of model "ml" if machine learning is to be used and "nn" if neural networks are to be used' )
        parser.add_argument('-i', '--input', type=list, help='Provide the input data for prediction', )
        

    def handle(self, *args, **kwargs):
        global dataset_path, config_path,  model_type
        dataset_path = kwargs['dataset']
        config_path = kwargs['config']
        model_type = kwargs['model_type']
        input_data = kwargs['input']       
        
        data = pd.read_csv(dataset_path)
        Preprocessor(data)
        if 'input' in input_data:
            predict(input_data)
        shutil.make_archive(model_config, 'zip',config_path)
        return dataset_path, config_path,  model_type  





def config(dir):  
    global response,model
    with open(dir,"r") as f:
        data = json.load(f)
    if data['model_type'] == 'nn':
        if data['model_name'] == 'CNN':
            model= CNN()
        elif data['model_name'] == 'RNN':
            model = RNN()
        elif data['model_name'] == 'Autoencoder':
            model = Autoencoder()
        else:
            response = 'Model doesn\'t exist'

    elif data['model_type'] == 'ml':
        if data['model_name'] == 'DecisionTree':
            model = DecisionTree()
        elif data['model_name'] == 'RandomForest':
            model = RandomForest()
        elif data['model_name'] == 'KNeighbors':
            model = KNeighbors()
        elif data['model_name'] == 'GaussianProcess':
            model = GaussianProcess()
        elif data['model_name'] == 'AdaBoost':
            model = AdaBoost()
        elif data['model_name'] == 'MLP':
            model = MLP()
        else:
            response = 'Model doesn\'t exist'
    else:
        response = 'Model Type not choosen'

    model.train()
    model.save_model(model, config_path)
    zip_file = zipfile.ZipFile("/local/my_files/my_file.zip", "w")

    return model, zip_file








#@api_view(['GET'])
def predict(data ):
    global model
    config(config_path)    
    
    if data['model_type'] == 'nn': 
        model = torch.load(config_path)
        _data = dataset.encode(input_data)
        _data = torch.from_numpy(np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)).reshape(-1, 1, 8, 8).cuda()
        _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
        response = dataset.decode(_out, label=True)    
        
    elif data['model_type'] == 'ml': 
        model = load_model(config_path)
        response = model.predict()
    else:
        response = 'Model Type not choosen'
        
    output = {'prediction': response}

    return JsonResponse(output)


