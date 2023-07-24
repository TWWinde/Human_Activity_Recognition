# Human Activity Recognition
This repository contains the code for training and evaluation of Deep Learning Lab work.

## Links

* [Project Poster](https://github.com/TWWinde/Diabetic_Retinopathy/blob/main/Diabetic_Retinopathy_Detection_based_on_Deep_Learning.pdf)
  
## Train and Evaluation

### 1. Datasets

Pre-process the [*IDRID Dataset*](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) using the code-base at `Input_pipeline/` (https://github.com/TWWinde/Diabetic_Retinopathy/tree/main/diabetic_retinopathy/Input_pipeline).
1. Edit the paths defined at the config script to point to the actual data paths. 
2. Run the script using `python data_prepare.py` to generate the needed TFrecord files.
   
       cd /Input_pipline
   
       python data_prepare.py
   
After the dataset preprocessing procedures have been performed, we can move on to the next steps.

### 2. Prerequisites

This codebase should run on most standard Linux systems. We specifically used Ubuntu 

Please install the following prerequisites manually (as well as their dependencies), by following the instructions found below:
* Tensorflow 

The remaining Python package dependencies can be installed by running:

       pip3 install --user --upgrade -r requirements.txt




# How to run the code

- Change  `batch.sh` to `python3 main.py` Run `main.py` to train or evaluation. Run `main.py` ,it will process the image and serialize images and labels into the TFRecord format and then training the model automatically.      
- Change  `batch.sh` to `python3 tune.py`. Run `tune.py` to do hyperparameter optimization. 



## Results of Human Activity Recognition
The GRU model finally achieved an accuracy of 95.82%, while the LSTM model finally achieved an accuracy of 94,40%.
### Performance
| Model | Train Accuracy [%] | Val Accuracy [%] | Test Accuracy [%] | Total Params |
|:-----:|:------------------:|:----------------:|:-----------------:|:------------:|
| LSTM  |       95.21        |      82.55       |       94.40       |    61,068    |
|  GRU  |       97.34        |      82.80       |       95.82       |    94,412    |

### LSTM
The model uses the lstm layer as the rnn layer. It is run with `python3 main.py --model_name lstm`.
The optimal parameters obtained by manual optimization are commented in the config-file.
### GRU
The model uses the gru layer as the rnn layer. It is run with `python3 main.py --model_name gru`.
The optimal parameters obtained by manual optimization are commented in the config-file.

### Confusion matrix
In the evaluation, the results of the trained model for the test set are presented as a confusion matrix. The output will be stored in the output of the trained model.
Below is the confusion matrix of the GRU model
### Confusion matrix Example 
![cm_gru](https://media.github.tik.uni-stuttgart.de/user/3535/files/f6061826-83a8-4653-bdba-c1bb039aeca2)

### Visualization
After the evaluation, a selected file will be visualized. The file to be visualized can be set in *config.gin* by editing the parameter `Visualization.exp_user = (2, 1)`.
This will create the visualization of acc_exp02_user01.txt & gyro_exp02_user01.txt.

The output will be stored in the output of the model that has been trained.
The visualization contains the ground truth and prediction of the accelerometer and gyroscope signals and their labels. Below is an example of a trained GRU model.

#### Visualization Example 
<img width="703" alt="visu" src="https://media.github.tik.uni-stuttgart.de/user/3535/files/9e91c960-c4c6-4029-98d0-59650df93666">


### Conclusions
- Both models have achieved high accuracy, which proves that the RNN networks can be applied for human activity recognition and have good performance.
- Due to the shorter duration of dynamic activities(some are even shorter than window length), there are less training data for them, which leads to a imbalanced data set. so the accuracy of dynamic activities is much lower than static one.

