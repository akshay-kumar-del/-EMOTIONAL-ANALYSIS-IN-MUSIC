from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import cv2


from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from xgboost import XGBClassifier #load ML classes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample


accuracy = []
precision = []
recall = []
fscore = []

model_folder = "Model"

def Upload_Dataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    
    # Feature extraction
    features = []
    
    # 1. MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc_mean = np.mean(mfcc, axis=1)
    features.extend(mfcc_mean)
      
    return np.array(features)
    
def load_dataset(X_file,Y_file,model_folder,dataset_path):
    X = []
    y = []
    for subfolder in os.listdir(dataset_path):          
        class_label = subfolder
        subfolder_path = os.path.join(dataset_path, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(subfolder_path, file)
                print(file_path)
                features = extract_features(file_path)
                X.append(features)
                y.append(class_label)
    np.save(X_file, X)
    np.save(Y_file, y)
    return np.array(X), np.array(y)

def Preprocess_Dataset():
    global X,Y,filename
    filepath = Path(filename)
    last_folder=filepath.name
 
    
    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
    else:
        X, Y = load_dataset(X_file,Y_file,model_folder,last_folder)
    
    desired_samples = 3000
    X, Y = resample(X, Y, n_samples=desired_samples, random_state=42)
    
    text.insert(END, "Preprocessing and MFCC Feature Extraction completed on Dataset: " + str(filename) + "\n\n")
    text.insert(END, "Input MFCC Feature Set Size: " + str(X.shape) + "\n\n")


def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test

    # Create a count plot


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def existing_classifier():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = SVC(probability=True)
    mlmodel.fit(x_train, y_train)
    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing SVM", y_pred, y_test)


def existing_classifier1():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = KNeighborsClassifier(n_neighbors=500)
    mlmodel.fit(x_train, y_train)
    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing KNN", y_pred, y_test)
    

def existing_classifier2():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = DecisionTreeClassifier(criterion = "entropy",max_leaf_nodes=2,max_features="auto")
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing DTC", y_pred, y_test)

    

def existing_classifier3():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = LogisticRegression(C=0.01, penalty='l1',solver='liblinear')
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing LRC", y_pred, y_test)

        

def existing_classifier4():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    #now train LSTM algorithm
 
    mlmodel = AdaBoostClassifier()
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing AdaBoost Classifier", y_pred, y_test)


    
def existing_classifier5():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    #now train LSTM algorithm
 
    mlmodel = LinearDiscriminantAnalysis()
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing Linear Discriminant Analysis", y_pred, y_test)


def proposed_classifier():
    global x_train,y_train,x_test,y_test,mlmodel
    text.delete('1.0', END)

    mlmodel = LGBMClassifier()
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Proposed Light Gradiant Boosting Classifier", y_pred, y_test)


def Prediction():
    global mlmodel, categories

    # File selection dialog
    filename = filedialog.askopenfilename(initialdir="Test Data")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')

    # Feature extraction
    features = extract_features(filename)
    features = features.reshape(1, -1)
    # Perform prediction
    predict = mlmodel.predict(features)  
    predicted_category = predict[0]  
    text.insert(END, f"Predicted Outcome From Test Audio is: {predicted_category}\n\n")

    # Load audio file
    y, sr = librosa.load(filename, sr=None)

    # Create a waveplot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Save plot to an image buffer
    plt.savefig("waveplot.png")
    plt.close()

    # Load the saved waveplot image
    waveplot_img = cv2.imread("waveplot.png")
    if waveplot_img is not None:
        # Add the predicted category text on the waveplot
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # Position to place the text
        font_scale = 1
        font_color = (0, 0, 255)  # Red color in BGR
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(waveplot_img, f"Predicted: {predicted_category}", position, font, font_scale, font_color, thickness, line_type)

        # Display the waveplot with annotation
        cv2.imshow("Waveplot with Prediction", waveplot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def close():
    main.destroy()


main = tkinter.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

font = ('times', 18, 'bold')
title = Label(main, text="Exploring emotional analysis in music for insights from the deam dataset and open smile features")
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
Button1 = Button(main, text="Upload Dataset", command=Upload_Dataset)
Button1.place(x=20,y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Preprocess Dataset", command=Preprocess_Dataset)
Button1.place(x=20,y=150)
Button1.config(font=font1)

Button1 = Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
Button1.place(x=20,y=200)
Button1.config(font=font1) 


Button1 = Button(main, text="Support Vector Machine", command=existing_classifier)
Button1.place(x=20,y=250)
Button1.config(font=font1)

Button1 = Button(main, text="K Nearest Neighbour", command=existing_classifier1)
Button1.place(x=20,y=300)
Button1.config(font=font1)

Button1 = Button(main, text="Decision Tree Classifier", command=existing_classifier2)
Button1.place(x=20,y=350)
Button1.config(font=font1)


Button1 = Button(main, text="Logistic Regression Classifier", command=existing_classifier3)
Button1.place(x=20,y=400)
Button1.config(font=font1)

Button1 = Button(main, text="AdaBoost Classifier", command=existing_classifier4)
Button1.place(x=20,y=450)
Button1.config(font=font1)

Button1 = Button(main, text="Linear Discriminant Analysis", command=existing_classifier5)
Button1.place(x=20,y=450)
Button1.config(font=font1)

Button1 = Button(main, text="Proposed LGBM", command=proposed_classifier)
Button1.place(x=20,y=500)
Button1.config(font=font1)

Button1 = Button(main, text="Prediction", command=Prediction)
Button1.place(x=20,y=550)
Button1.config(font=font1)


Button1 = Button(main, text="Exit", command=close)
Button1.place(x=20,y=600)
Button1.config(font=font1)



                            
font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=95)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300,y=100)
text.config(font=font1)
main.config(bg='SeaGreen1')

main.mainloop()
