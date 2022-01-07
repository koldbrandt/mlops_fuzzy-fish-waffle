# mlops_fuzzy-fish-waffle
The dataset is found on kaggle called ["A Large Scale Fish Dataset"](https://www.kaggle.com/crowww/a-large-scale-fish-dataset). 
We want to utilize kornia to do image augmentation and to train the network. The model will be a simple convolutional neural network to classify the 9 different kind of fish. 

The augmentation of the images will be to crop the images to focus on the center. To flip the images and rotate them since pictures of fish could potentially be in all directions. The images will also be normalized. 

The model will have 3 convolutional layers with batchnormalization and relu will be used as activation function. The images will be flatten and fed into a dense hidden layer that continues into the classification layer. This will hopefully give an accuracy higher than 97%. Kornia will also be used to create the network and train it. 


### Week 1
- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages (using conda)
- [x] Create the initial file structure using cookiecutter
- [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
- [ ] Add a model file and a training script and get that running
- [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (`pep8`) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use wandb to log training progress and other important metrics/artifacts in your code
- [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code



