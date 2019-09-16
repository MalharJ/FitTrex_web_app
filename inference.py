# Import basic libraries
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def softmax(x):
    maxval = torch.max(x)
    denom = torch.sum(torch.exp(x - maxval))
    num = torch.exp(x - maxval)
    return (num/denom).detach().numpy()

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

if __name__ == '__main__':
    
    # Load the parser that can parse arguments
    parser = ArgumentParser()
    
    # Add the arguments. The arguments we want specifically are: 
    # (1) Path to the model dictionary file
    # (2) Path to the labels for the model
    # (3) Path to the image file to load
    parser.add_argument("-model", "--model_dict", dest="model_dictionary_path",
                    help="specify path to the model dictionary file", metavar="Path to the model dictionary file")
    
    parser.add_argument("-label", "--labels", dest="model_labels_path",
                    help="specify path to the labels for the model", metavar="Path to the labels for the model")
    
    parser.add_argument("-img", "--image", dest="imagepath_for_inference",
                    help="specify path to the image file for inference", metavar="Path to the image file to load")
    
    args = parser.parse_args()
    
    model_dictionary_path = str(args.model_dictionary_path)
    model_labels_path = str(args.model_labels_path)
    imagepath_for_inference = str(args.imagepath_for_inference)
    
    # Predefine the model parameters. Think of it like loading a body into a skeleton.
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)

    # Now we load the actual model from a file
    model.load_state_dict(torch.load(model_dictionary_path, map_location='cpu'))
    model.eval()

    # Let's load the labels for the model
    labels = [line.rstrip('\n') for line in open(model_labels_path)]

    # Let's load in the image for inference and do processing on it for the model
    image = Image.open(imagepath_for_inference)
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    # Inference time
    inference_vector = model.forward(batch_t)
    top_class = np.argmax(softmax(inference_vector))
    top_probability = np.max(softmax(inference_vector))
    
    # Print the result
    print (imagepath_for_inference, labels[top_class], top_probability)