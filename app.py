# import os
# from flask import Flask, send_file, request, jsonify
# from werkzeug.exceptions import BadRequest
# from werkzeug.utils import secure_filename
# import torch
# from torchvision import datasets, models, transforms
# import torch.nn as nn
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# from argparse import ArgumentParser

# app = Flask(__name__)

def softmax(x):
    maxval = torch.max(x)
    denom = torch.sum(torch.exp(x - maxval))
    num = torch.exp(x - maxval)
    return (num/denom)

# from torchvision import transforms
# transform = transforms.Compose([            #[1]
#  transforms.Resize(256),                    #[2]
#  transforms.CenterCrop(224),                #[3]
#  transforms.ToTensor(),                     #[4]
#  transforms.Normalize(                      #[5]
#  mean=[0.485, 0.456, 0.406],                #[6]
#  std=[0.229, 0.224, 0.225]                  #[7]
#  )])

# def load_model(model_dictionary_path):
#     """Load and return the model"""
#     # TODO: INSERT CODE
#     # return model
#     model = models.resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 101)
#     model.load_state_dict(torch.load(model_dictionary_path, map_location='cpu'))
#     model.eval()
#     return model



# # you can then reference this model object in evaluate function/handler
# model = load_model(model_dictionary_path='resnet18_food101_state_dict')


# # The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
# @app.route('/', methods=['GET', "POST"])
# def evaluate():
#     """Preprocessing the data and evaluate the model"""
#     # TODO: data/input preprocessing
#     # eg: request.files.get('file')
#     # eg: request.args.get('style')
#     # eg: request.form.get('model_name')

#     # TODO: model evaluation
#     # eg: prediction = model.eval()

#     # TODO: return prediction
#     # eg: return jsonify({'score': 0.95})

#     print (request.data)

#     input_file = request.files.get('file')

#     print (type(input_file))

#     if not input_file:
#         print ('Bad request')
#         return BadRequest("File not present in request")

#     filename = secure_filename(input_file.filename)
#     if filename == '':
#         return BadRequest("File name is not present in request")
#     if not allowed_file(filename):
#         return BadRequest("Invalid file type")

#     labels = [line.rstrip('\n') for line in open('food101_labels.txt')]

#     print (input_file)

#     return type(input_file)



# # The following is for running command `python app.py` in local development, not required for serving on FloydHub.
# if __name__ == "__main__":
#     print("* Starting web server... please wait until server has fully started")
#     app.run(host='0.0.0.0', threaded=False)

import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, send_file, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from argparse import ArgumentParser

app = Flask(__name__)
# imagenet_class_index = [line.rstrip('\n') for line in open('food101_labels.txt')]

# def load_model():
#     model = models.resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 101)
#     model.load_state_dict(torch.load('resnet18_food101_state_dict', map_location='cpu'))
#     model.eval()

def data():
    incoming = request.form.get('query')
    print(incoming)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))

    intermediate = image.convert('RGB')

    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%  S")

    today = str(dt_string)
    image.save(today + '.bmp')

    intermediate_ = my_transforms(intermediate)

    return intermediate_.unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)
    model.load_state_dict(torch.load('resnet18_food101_state_dict', map_location='cpu'))
    model.eval()
    imagenet_class_index = [line.rstrip('\n') for line in open('food101_labels.txt')]

    outputs = model.forward(tensor)

    top_10_probs_, top_10_indices_ = outputs.topk(10)

    top_10_indices = []
    for val in top_10_indices_[0]:
        top_10_indices.append(int(val.numpy()))

    top_10_labels = [imagenet_class_index[j] for j in top_10_indices]

    top_10_probs = []
    for val in softmax(top_10_probs_[0]).detach().numpy():
        top_10_probs.append(float(val))

    print (top_10_probs, type(top_10_probs[0]))

    return top_10_indices, top_10_labels, top_10_probs

    # predicted_idx = torch.argmax(outputs[0,:])

    # return int(predicted_idx), imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print ('why hello there')

        calorie_information = pd.read_csv('food101_nutritional_info.csv')

        file = request.files['file']

        print ('EVERYTHANG', request.form)

        img_bytes = file.read()
        class_ids, class_names, class_probs = get_prediction(image_bytes=img_bytes)

        for i in range(10):
            print ('id: ', class_ids[i], ', name: ', class_names[i], ', probability: ', class_probs[i])

        # Need to return pandas rows in json format where the indices are class ids
        protein_weight = []
        carbohydrate_weight = []
        fat_weight = []
        calories = []
        weight_per_serving = []

        for val in class_ids:
            row  = (calorie_information.loc[calorie_information['Index'] == val])
            print (type(row['n_protein_weight']), type(row['n_carb_weight']), type(row['n_fat_weight']), type(row['n_calories']), 
                type(row['weight']))

            print (row['n_protein_weight'], row['n_carb_weight'], row['n_fat_weight'], row['n_calories'], row['weight'])

            # protein_weight.append(float(row['protein_weight']))
            # carbohydrate_weight.append(float(row['carb_weight']))
            # fat_weight.append(float(row['fat_weight']))
            # calories.append(float(row['calories']))
            # weight_per_serving.append(float(row['weight']))

            protein_weight.append(float(row['n_protein_weight']))
            carbohydrate_weight.append(float(row['n_carb_weight']))
            fat_weight.append(float(row['n_fat_weight']))
            calories.append(float(row['n_calories']))
            weight_per_serving.append(100)

        return jsonify(ids=class_ids, probs=class_probs, result=class_names, protein_weights=protein_weight, 
                       carbohydrate_weight=carbohydrate_weight, fat_weight=fat_weight, calories=calories, weight_per_serving=weight_per_serving)

if __name__ == '__main__':
    print ('wtf')
    app.run()