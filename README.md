# Kera_to_MDF-Project

## DESCRIPTION
This Project is for the Task 2 of the ModECI MDF December 2022 Outreachy Internship Cohort. This project is a tutorial on how to creating MDF models with Keras.
For this project i will be creating a simple step-by-step keras model then I will convert it into MDF model.

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

Also, for this project I created a new python virtual environment. This can be with the following code, project directly.
```
python -m venv (project directory)

```
This project was built with [VsCode](https://code.visualstudio.com/).

## DATASET
The dataset I used for this project is the **PIMA INDIA DIABETES DATASET** that I got from [kaggle](https://www.kaggle.com/). The dataset comprises of 9 columns which I later convert to int(0 to 8), they includes;

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome

It also comprise of 768 rows.


