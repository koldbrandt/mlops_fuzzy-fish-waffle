# mlops_fuzzy-fish-waffle
The dataset is found on kaggle called ["A Large Scale Fish Dataset"](https://www.kaggle.com/crowww/a-large-scale-fish-dataset). 
We want to utilize kornia to do image augmentation and to train the network. The model will be a simple convolutional neural network to classify the 9 different kind of fish. 

The augmentation of the images will be to crop the images to focus on the center. To flip the images and rotate them since pictures of fish could potentially be in all directions. The images will also be normalized. 

The model will have 3 convolutional layers with batchnormalization and relu will be used as activation function. The images will be flatten and fed into a dense hidden layer that continues into the classification layer. This will hopefully give an accuracy higher than 97%. We will start by creating this model using pytorch. If there is more time a model will be created using kornia. 


### Week 1
- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages (using conda)
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
- [x] Add a model file and a training script and get that running
- [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (`pep8`) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [x] Setup version control for your data or part of your data
- [x] Construct one or multiple docker files for your code
- [x] Build the docker files locally and make sure they work as intended
- [x] Write one or multiple configurations files for your experiments
- [x] Used Hydra to load the configurations and manage your hyperparameters
- [x] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [x] Use wandb to log training progress and other important metrics/artifacts in your code
- [x] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [x] Write unit tests related to the data part of your code
- [x] Write unit tests related to model construction
- [x] Calculate the coverage.
- [x] Get some continuous integration running on the github repository
- [X] (optional) Create a new project on `gcp` and invite all group members to it
- [X] Create a data storage on `gcp` for you data
- [X] Create a trigger workflow for automatically building your docker images
- [X] Get your model training on `gcp`
- [x] Play around with distributed data loading
- [ ] (optional) Play around with distributed model training
- [ ] Play around with quantization and compilation for you trained models

### Week 3
- [x] Deployed your model locally using TorchServe
- [x] Checked how robust your model is towards data drifting
- [ ] Deployed your model using `gcp`
- [ ] Monitored the system of your deployed model
- [ ] Monitored the performance of your deployed model
