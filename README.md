# Keras _to_MDF-Project

## DESCRIPTION
This Project is for the ModECI MDF Outreachy December 2022 cohort contribution phase Task 2 project. This project is a tutorial on how to implement MDF with Keras, i will be creating a simple step-by-step keras model that will be convert into MDF model.

## REQUIREMENT
I will be using some libraries to get this done and the include;
1. [tensorflow](https://www.tensorflow.org/)-This is where I will be importing Keras from
2. [pandas](https://pandas.pydata.org/)
3. [numpy](https://numpy.org/)
4. [sklearn](https://scikit-learn.org/)
5. [scipy](https://scipy.org/)
6. [modeci_mdf](https://pypi.org/project/modeci-mdf/)

All these above can be downloaded via
```
pip install -r requirement.txt

```

Also, for this task I created a new python virtual environment. This can be done with the following code, followed by the project directory. 
```
python -m venv (project directory)

```
This tutorial was created with [VsCode](https://code.visualstudio.com/) and [jupyter_notebook](https://jupyter.org/)

## DATASET
The dataset I used for this project is the **PIMA INDIA DIABETES DATASET** that I got from [kaggle](https://www.kaggle.com/). The dataset comprises of 9 columns which I later convert to integers(0 to 8) respectively, they includes;

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome

The dataset also comprise of 768 rows.


## Workflow
1. I created a simple keras model using the diabetes dataset on jupyter which was saved to the h5 format.
2. The keras model saved to h5 was imported to python script, and various functions to convert the keras model to Mdf project was implemented in this script. 
3. In the last workflow of this project, the keras model is finally converted to mdf project with the aid of the previously created functions on the python script. The newly mdf model is saved in both json and yaml files, the graphical representation of the model is also created, and lastly the performance of the mdf project is evaluated.This is based on a jupyter notebook file.

### Thanks for reading!!!