import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import sklearn
from keras.optimizers import SGD

# acteur=[]
# with open("noteacteur_final.txt") as fp:
#     line = fp.readline()
#     while line:
#         act=list()
#         data = line.split(';')
#         if len(data) >0:
#             act.append(data[0])
#             act.append(data[1])
#             act.append(data[2])
#             act.append(data[3])
#             act.append(data[4])
#             act.append(data[5])
#             act.append(data[6])
#             act.append(float(data[7]))
#             acteur.append(act)
#         line = fp.readline()
#
# realisateur=[]
# with open("noterealisateur_final.txt") as fp:
#     line = fp.readline()
#     while line:
#         act=list()
#         data = line.split(';')
#         if len(data) >0:
#             act.append(data[0])
#             act.append(data[1])
#             act.append(data[2])
#             act.append(data[3])
#             act.append(data[4])
#             act.append(data[5])
#             act.append(float(data[6]))
#             realisateur.append(act)
#         line = fp.readline()
# actrice=[]
# with open("noteactrice_final.txt") as fp:
#     line = fp.readline()
#     while line:
#         act=list()
#         data = line.split(';')
#         if len(data) >0:
#             act.append(data[0])
#             act.append(data[1])
#             act.append(data[2])
#             act.append(data[3])
#             act.append(data[4])
#             act.append(data[5])
#             act.append(data[6])
#             act.append(float(data[7]))
#             actrice.append(act)
#         line = fp.readline()
#
# Datasetfilms=[]
# with open("TestDATA-copie.txt") as fp:
#     line = fp.readline()
#     while line:
#         act=list()
#         data = line.split(';')
#         if len(data) >0:
#             act.append(data[0])
#             act.append(data[1])
#             act.append(data[2])
#             act.append(data[3])
#             act.append(data[4])
#             act.append(data[5])
#             act.append(data[6])
#             act.append(data[7])
#             act.append(data[8])
#             act.append(data[9])
#             act.append(data[10])
#             act.append(data[11])
#             act.append(data[12])
#             act.append(data[13])
#             Datasetfilms.append(act)
#         line = fp.readline()
# for film in Datasetfilms:
#     for act in acteur:
#         for acti in actrice:
#             for rea in realisateur:
#                 if (film[2]==act[1]):
#                     film[9]=act[7]
#                 if (film[3]==act[1]):
#                     film[10]=act[7]
#                 if (film[4]==acti[1]):
#                     film[11]=acti[7]
#                 if (film[5]==acti[1]):
#                     film[12]=acti[7]
#                 if (film[6]==rea[0]):
#                     film[13]=rea[6]
#
# for f in Datasetfilms:
#     fichier = open("TestDATA.txt", "a")
#     fichier.write(f[0]+';'+f[1]+';'+f[2]+';'+f[3]+';'+f[4]+';'+f[5]+';'+f[6]+';'+f[7]+';'+f[8]+';'+str(f[9])+';'+str(f[10])+';'+str(f[11])+';'+str(f[12])+';'+str(f[13])+';\n')
#     fichier.close()
def deeplearning():
    movies = pd.read_csv('movieDATA.csv',sep=',')
    movies=movies.drop(['Unnamed: 14'], axis=1)
    movies=movies.drop(['Id'], axis=1)
    movies=sklearn.utils.shuffle(movies)
    moviestest = pd.read_csv('movieTestDATA.csv',sep=',')
    moviestest=moviestest.drop(['Unnamed: 14'], axis=1)
    moviestest=moviestest.drop(['Id'], axis=1)
    # print(movies.head())
    X=movies.iloc[:,8:]
    Y=movies.iloc[:,7:8]
    X=X.values
    Y=Y.values
    Y=to_categorical(Y)
    movieTestX=moviestest.iloc[:,8:]
    movieTestX=movieTestX.values
    trainX , testX = X[:63 , :], X[63:, :]
    trainY , testY = Y[:63], Y[63:]
    # print(trainY.shape)
    model=keras.Sequential()
    model.add(keras.layers.Dense(50, activation='relu', input_shape=(5,))) #création du modèlechoix du neural networl avec 2 couches
    model.add(keras.layers.Dense(3, activation='softmax'))
    # model.summary()
    lrate =0.01
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate= lrate),
              metrics =['accuracy'])
    history = model.fit(trainX , trainY ,
                  validation_data =(testX , testY),
                  epochs=200,verbose=0)
    prediction=model.predict(movieTestX)
    pred=list()
    # print(len(prediction))
    for i in range(0,16,1):
        pro=list()
        for j in range(0,3,1):
            pro.append(prediction[i][j])
        pred.append(pro)
    for i in range(0,16,1):
        pred[i].append(moviestest["Title"][i])
    retour=""
    for pre in pred:
        retour+=pre[3]+' est un: Hit: '+str(pre[2])+', Average: '+str(pre[1])+', Flop: '+str(pre[0])+'.\n'
        # print(pre[3]+' est un: Hit:'+str(pre[2])+', Average:'+str(pre[1])+', Flop:'+str(pre[0]))
    return retour

deeplearning()