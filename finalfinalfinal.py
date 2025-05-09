#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries from matplotlib to visualize the data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Importing Libraries to create GUI
from tkinter import *

#Importing Libraries to perform calculations
import numpy as np
import pandas as pd
import os


# In[2]:


#pip install mysql-connector-python. This should run in the terminal not in the script
import mysql.connector  # Import MySQL connector after installing it



# In[3]:


# list of symptoms 

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']


# In[4]:


# list of disease

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae',
         'AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',' Migraine','Cervical spondylosis',
         'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
         'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
         'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)', 'Heartattack','Varicoseveins','Hypothyroidism'
         'Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo',
         'Acne','Urinary tract infection','Psoriasis','Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)
print(l2)


# In[10]:


df = pd.read_csv("Training.csv")

df.replace({'prognosis': {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
    'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15,
    'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
    'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26,
    'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
    'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39,
    'Impetigo': 40
}}, inplace=True)

# ✅ Add this line to fix the warning
df = df.infer_objects(copy=False)

# print(df.head())

X = df[l1]
y = df[["prognosis"]]
y = np.ravel(y)  # ✅ Assign the result back to 'y' to avoid unnecessary computation

# print(y)


# # training data

# In[9]:


tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


# In[11]:


#list1 = DF['prognosis'].unique()
def scatterplt(disea):
    x = ((DF.loc[disea]).sum())#total sum of symptom reported for given disease
    x.drop(x[x==0].index,inplace=True)#droping symptoms with values 0
    print(x.values)
    y = x.keys()#storing nameof symptoms in y
    print(len(x))
    print(len(y))
    plt.title(disea)
    plt.scatter(y,x.values)
    plt.show()
def scatterinp(sym1,sym2,sym3,sym4,sym5):
    x = [sym1,sym2,sym3,sym4,sym5]#storing input symptoms in y
    y = [0,0,0,0,0]#creating and giving values to the input symptoms
    if(sym1!='Select Here'):
        y[0]=1
    if(sym2!='Select Here'):
        y[1]=1
    if(sym3!='Select Here'):
        y[2]=1
    if(sym4!='Select Here'):
        y[3]=1
    if(sym5!='Select Here'):
        y[4]=1
    print(x)
    print(y)
    plt.scatter(x,y)
    plt.show()


# # decision tree
# 

# In[12]:


root = Tk()
pred1=StringVar()
def DecisionTree():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif len(AddressEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Address")
        if comp:
            root.mainloop()   
    elif len(MobileEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Contact details")
        if comp:
            root.mainloop()         
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here") or (Symptom3.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first three Symptoms")
        if sym:
            root.mainloop()
    else:
        print(NameEn.get())
        from sklearn import tree

        clf3 = tree.DecisionTreeClassifier() 
        clf3 = clf3.fit(X,y)

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=clf3.predict(X_test)
        print("Decision Tree")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf3.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break


        if (h=='yes'):
            pred1.set(" ")
            pred1.set(disease[a])
        else:
            pred1.set(" ")
            pred1.set("Not Found")

        import mysql.connector;
        conn = mysql.connector.connect(host='localhost',user='root',password='Jai,12345',database='pirojaikt');
        mycursor = conn.cursor();
        mycursor.execute("CREATE TABLE IF NOT EXISTS DecisionTree(Name char(20), Address varchar(20), Email varchar(30),Mobile varchar(50), Symptom1 char(30),Symptom2 char(30),Symptom3 char(30),Symptom4 char(30),Symptom5 char(30),Disease char(30))");
        mycursor.execute("INSERT INTO DecisionTree(Name,Address,Email,Mobile,Symptom1,Symptom2,Symptom3,Symptom4,Symptom5,Disease) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(NameEn.get(),AddressEn.get(),EmailEn.get(),MobileEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred1.get()));
        conn.commit()  
        mycursor.close(); 
        conn.close();





# # random forest

# In[8]:


pred2=StringVar()
def randomforest():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif len(AddressEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Address")
        if comp:
            root.mainloop()   
    elif len(MobileEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Contact details")
        if comp:
            root.mainloop()         
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")or (Symptom3.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first three Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf4 = RandomForestClassifier(n_estimators=100)
        clf4 = clf4.fit(X,np.ravel(y))

        # calculating accuracy 
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=clf4.predict(X_test)
        print("Random Forest")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred2.set(" ")
            pred2.set(disease[a])
        else:
            pred2.set(" ")
            pred2.set("Not Found")

        import mysql.connector;
        conn = mysql.connector.connect(host='localhost',user='root',password='Jai,12345',database='pirojaikt');
        mycursor = conn.cursor();
        mycursor.execute("CREATE TABLE IF NOT EXISTS randomforest(Name char(20), Address varchar(20), Email varchar(30),Mobile varchar(50), Symtom1 char(30),Symtom2 char(30),Symtom3 char(30),Symtom4 char(30),Symtom5 char(30),Disease char(30))");
        mycursor.execute("INSERT INTO randomforest(Name,Address,Email,Mobile,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(NameEn.get(),AddressEn.get(),EmailEn.get(),MobileEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred2.get()));
        conn.commit()  
        mycursor.close(); 
        conn.close();


# # k nearest neighbour

# In[9]:


pred4=StringVar()
def KNN():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif len(AddressEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Address")
        if comp:
            root.mainloop()   
    elif len(MobileEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Contact details")
        if comp:
            root.mainloop()         
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")or (Symptom3.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first three Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        knn=knn.fit(X,np.ravel(y))

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=knn.predict(X_test)
        print("KNN")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)



        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = knn.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break


        if (h=='yes'):
            pred4.set(" ")
            pred4.set(disease[a])
        else:
            pred4.set(" ")
            pred4.set("Not Found")

        import mysql.connector;
        conn = mysql.connector.connect(host='localhost',user='root',password='Jai,12345',database='pirojaikt');
        mycursor = conn.cursor();
        mycursor.execute("CREATE TABLE IF NOT EXISTS knearest(Name char(20), Address varchar(20), Email varchar(30),Mobile varchar(50), Symtom1 char(30),Symtom2 char(30),Symtom3 char(30),Symtom4 char(30),Symtom5 char(30),Disease char(30))");
        mycursor.execute("INSERT INTO knearest(Name,Address,Email,Mobile,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(NameEn.get(),AddressEn.get(),EmailEn.get(),MobileEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred3.get()));
        conn.commit()  
        mycursor.close(); 
        conn.close();


# # naive byes

# In[10]:


pred3=StringVar()
def NaiveBayes():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif len(AddressEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Address")
        if comp:
            root.mainloop()   
    elif len(MobileEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Contact details")
        if comp:
            root.mainloop()          
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")or (Symptom3.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first three Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb=gnb.fit(X,np.ravel(y))

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=gnb.predict(X_test)
        print("Naive Bayes")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = gnb.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred3.set(" ")
            pred3.set(disease[a])
        else:
            pred3.set(" ")
            pred3.set("Not Found")


        import mysql.connector;
        conn = mysql.connector.connect(host='localhost',user='root',password='Jai,12345',database='pirojaikt');
        mycursor = conn.cursor();
        mycursor.execute("CREATE TABLE IF NOT EXISTS naivebyes(Name char(20), Address varchar(20), Email varchar(30),Mobile varchar(50), Symtom1 char(30),Symtom2 char(30),Symtom3 char(30),Symtom4 char(30),Symtom5 char(30),Disease char(30))");
        mycursor.execute("INSERT INTO naivebyes(Name,Address,Email,Mobile,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(NameEn.get(),AddressEn.get(),EmailEn.get(),MobileEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred3.get()));
        conn.commit()  
        mycursor.close(); 
        conn.close();


# # GUI creation using tkinter

# In[11]:


#Tk class is used to create a root window
root.configure(background='ivory')
root.title('Disease Prediction System')
#root.resizable(0,0)

#taking first input as symptom
Symptom1 = StringVar()
Symptom1.set("Select Here")

#taking second input as symptom
Symptom2 = StringVar()
Symptom2.set("Select Here")

#taking third input as symptom
Symptom3 = StringVar()
Symptom3.set("Select Here")

#taking fourth input as symptom
Symptom4 = StringVar()
Symptom4.set("Select Here")

#taking fifth input as symptom
Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()

Address = StringVar()

Email = StringVar()

Mobile = StringVar()


# In[12]:


#Reset Button



#function to Reset the given inputs to initial position
prev_win=None
def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")

    NameEn.delete(first=0,last=100)

    AddressEn.delete(first=0,last=100)

    EmailEn.delete(first=0,last=100)

    MobileEn.delete(first=0,last=100)


    pred1.set(" ")
    pred2.set(" ")
    pred3.set(" ")
    pred4.set(" ")
    try:
        prev_win.destroy()
        prev_win=None
    except AttributeError:
        pass



# In[13]:


#Exit Button


#Exit button to come out of system
from tkinter import messagebox
def Exit():
    qExit=messagebox.askyesno("System","Do you want to exit the system")
    if qExit:
        root.destroy()
        exit()


#Headings at the Top of GUI

#Headings for the GUI written at the top of GUI
w2 = Label(root, justify=LEFT, text="Disease Prediction using Machine Learning", fg="Red", bg="ivory")
w2.config(font=("Times New Roman",18,"bold"))
w2.grid(row=1, column=0, columnspan=2, padx=60)
# w2 = Label(root, justify=LEFT, fg="Dark blue", bg="Ivory")
# w2.config(font=("Times New Roman",18,"bold"))
# w2.grid(row=2, column=0, columnspan=2, padx=60)


# In[ ]:


#Creating Labels for Patient Details



# labels for the name

NameLb = Label(root, text="Name of the Patient", fg="dark blue", bg="ivory")
NameLb.config(font=("Times New Roman",11,"bold"))
NameLb.grid(row=6, column=0, pady=8, sticky=W)

AddressLb = Label(root, text="Address of the Patient", fg="dark blue", bg="ivory")
AddressLb.config(font=("Times New Roman",11,"bold"))
AddressLb.grid(row=7, column=0, pady=8, sticky=W)

EmailLb = Label(root, text="Email of the Patient", fg="dark blue", bg="ivory")
EmailLb.config(font=("Times New Roman",11,"bold"))
EmailLb.grid(row=8, column=0, pady=8, sticky=W)

MobileLb = Label(root, text="Contact No. of the Patient", fg="dark blue", bg="ivory")
MobileLb.config(font=("Times New Roman",11,"bold"))
MobileLb.grid(row=9, column=0, pady=8, sticky=W)



#Creating Labels for symptoms



# Creating Labels for the symptoms

S1Lb = Label(root, text="Symptom 1", fg="Black", bg="Ivory")
S1Lb.config(font=("Times",11,"bold"))
S1Lb.grid(row=10, column=0, pady=8, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="Black", bg="Ivory")
S2Lb.config(font=("Times New Roman",11,"bold"))
S2Lb.grid(row=11, column=0, pady=8, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Black",bg="Ivory")
S3Lb.config(font=("Times New Roman",11,"bold"))
S3Lb.grid(row=12, column=0, pady=8, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Black", bg="Ivory")
S4Lb.config(font=("Times New Roman",11,"bold"))
S4Lb.grid(row=13, column=0, pady=8, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Black", bg="Ivory")
S5Lb.config(font=("Times New Roman",11,"bold"))
S5Lb.grid(row=14, column=0, pady=8, sticky=W)







#Creating Labels for ML algorithms
# Creating Labels for the ML Algorithms

lrLb = Label(root, text="DecisionTree", fg="white", bg="red", width = 20)
lrLb.config(font=("Times New Roman",12,"bold"))
lrLb.grid(row=17, column=0, pady=9,sticky=W)

destreeLb = Label(root, text="RandomForest", fg="WHITE", bg="DARK BLUE", width = 20)
destreeLb.config(font=("Times New Roman",12,"bold"))
destreeLb.grid(row=19, column=0, pady=9, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="White", bg="dark green", width = 20)
ranfLb.config(font=("Times New Roman",12,"bold"))
ranfLb.grid(row=21, column=0, pady=9, sticky=W)

knnLb = Label(root, text="kNearestNeighbour", fg="WHITE", bg="olive", width = 20)
knnLb.config(font=("Times New Roman",12,"bold"))
knnLb.grid(row=23, column=0, pady=9, sticky=W)
OPTIONS = sorted(l1)





#Taking Patient Personal Details
# Taking name as input from user
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

AddressEn = Entry(root, textvariable=Address)
AddressEn.grid(row=7, column=1)

EmailEn = Entry(root, textvariable=Email)
EmailEn.grid(row=8, column=1)

MobileEn = Entry(root, textvariable=Mobile)
MobileEn.grid(row=9, column=1)











#Taking symptoms as input from Patient
#Taking Symptoms as input from the dropdown from the user
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=10, column=1)

S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=11, column=1)

S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=12, column=1)

S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=13, column=1)

S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=14, column=1)












#Buttons for predicting the disease
# Buttons for predicting the disease using different Algorithms

dst = Button(root, text="Decision Tree", command=DecisionTree,bg="Red",fg="yellow")
dst.config(font=("Times New Roman",12,"bold"))
dst.grid(row=9, column=3,padx=8)

rnf = Button(root, text="Random Forest", command=randomforest,bg="dark blue",fg="light cyan")
rnf.config(font=("Times New Roman",12,"bold"))
rnf.grid(row=10, column=3,padx=8)

lr = Button(root, text="Naive Bayes", command=NaiveBayes,bg="dark green",fg="white")
lr.config(font=("Times New Roman",12,"bold"))
lr.grid(row=11, column=3,padx=8)

kn = Button(root, text="KNN", command=KNN,bg="olive",fg="silver")
kn.config(font=("Times New Roman",12,"bold"))
kn.grid(row=12, column=3,padx=8)

rs = Button(root,text="Reset Inputs", command=Reset,bg="chocolate",fg="ivory",width=15)
rs.config(font=("Times New Roman",12,"bold"))
rs.grid(row=13,column=3,padx=8)

ex = Button(root,text="Exit System", command=Exit,bg="dark red",fg="wheat",width=15)
ex.config(font=("Times New Roman",12,"bold"))
ex.grid(row=14,column=3,padx=8)


# In[15]:


#Output of different Algorithms used in this system






# showing the output of different Algorithms

t1=Label(root,font=("Times New Roman",10,"bold"),text="Decision Tree",height=1,bg="red"
         ,width=30,fg="white",textvariable=pred1,relief="sunken").grid(row=17, column=1, padx=8)

t2=Label(root,font=("Times New Roman",10,"bold"),text="Random Forest",height=1,bg="dark blue"
         ,width=30,fg="white",textvariable=pred2,relief="sunken").grid(row=19, column=1, padx=8)

t3=Label(root,font=("Times New Roman",10,"bold"),text="Naive Bayes",height=1,bg="dark green"
         ,width=30,fg="white",textvariable=pred3,relief="sunken").grid(row=21, column=1, padx=8)

t4=Label(root,font=("Times New Roman",10,"bold"),text="KNN",height=1,bg="olive"
         ,width=30,fg="white",textvariable=pred4,relief="sunken").grid(row=23, column=1, padx=8)


# In[ ]:


# CALLING the below function
root.mainloop()


# In[ ]:




