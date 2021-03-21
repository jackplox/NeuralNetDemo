from torchvision import models


##
#  Neural Nets

#network = models.alexnet(pretrained = True)
#network = models.resnet101(pretrained = True)
#network = models.squeezenet1_1(pretrained = True)
network = models.densenet201(pretrained = True)

from torchvision import transforms

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])])

from PIL import Image

##
#  Images

img = Image.open('labrador.jpg')
#img = Image.open('cow.jpg')
#img = Image.open('falcon.jpg')

img_t = preprocess(img)

import torch

batch_t = torch.unsqueeze(img_t, 0)

network.eval()

out = network(batch_t)

with open('imagenet1000_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
    
percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100
_, indices = torch.sort(out, descending = True)

for idx in range(5): 
    print(labels[indices[0][idx]] + ": " + str(percentage[indices[0][idx]]))
    
    


