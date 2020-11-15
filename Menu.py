from KNN import Film
from KNN import separate_by_class
from KNN import summarize_by_class
from KNN import calculate_class_probabilities
from KNN import predict
from KNN import naive_bayes
from KNN import KNN
from DeepLearningPTS import deeplearning
import NaiveBayes2 as NB2
from tkinter import *


#Creation de l'interface
class Interface(Frame):
    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, width=800, height=600, **kwargs)
        self.pack(fill=BOTH, expand=YES)

        # Creation de nos widgets
        
        self.message = Label(self, text="BIENVENUE SUR L'INTERFACE DE PRÉDICTION DU SUCCES D'UN FILM.\n\nNous avons implémenter 2 Naïve Bayes, un KNN et un réseau de neurones afin de prédire le succès d'un film. Vous pouvez chosir l'algo de votre choix : ", wraplength=500)
        self.message.pack( fill=BOTH, expand=YES)
        
        self.bouton_quitter = Button(self, text="Quitter", command=self.quit)
        self.bouton_quitter.pack(side="left", padx = 10, pady = 10, expand=YES)
            
        self.bouton_fonction1 = Button(self, text="Algo Bayes Classifer", fg="red", command=self.fonction1)
        self.bouton_fonction1.pack(side="left", padx =10, expand=YES)

        self.bouton_fonction2 = Button(self, text="KNN", fg="red", command=self.fonction2)
        self.bouton_fonction2.pack(side="left", padx = 10, expand=YES)

        self.bouton_fonction3 = Button(self, text="Deep Learning", fg="red", command=self.fonction3)
        self.bouton_fonction3.pack(side="left", padx = 10, expand=YES)

        self.bouton_fonction4 = Button(self, text="Naive Bayes 2", fg="red", command=self.fonction5)
        self.bouton_fonction4.pack(side="left", padx=10, expand=YES)

        self.bouton_fonction5 = Button(self, text="Résultat", fg="red", command=self.fonction4)
        self.bouton_fonction4.pack(side="left", padx = 10, expand=YES)

        


    def fonction1(self):
        Datasetfilms = []
        with open("TrainDATA.txt") as fp:
            line = fp.readline()
            while line:
                # print("o")
                film = Film()
                data = line.split(';')
                if len(data) > 0:
                    film.ID = data[0]
                    film.titre = data[1]
                    film.acteurs.append(data[2])
                    film.acteurs.append(data[3])
                    film.actrices.append(data[4])
                    film.actrices.append(data[5])
                    film.realisateur = data[6]
                    film.noteacteur1 = float(data[9])
                    film.noteacteur2 = float(data[10])
                    film.noteactrice1 = float(data[11])
                    film.noteactrice2 = float(data[12])
                    film.noterealisateur = float(data[13])
                    film.type = data[8]
                    Datasetfilms.append(film)
                line = fp.readline()
        Test = []
        with open("TestDATA.txt") as fp:
            line = fp.readline()
            while line:
                film = Film()
                data = line.split(';')
                if len(data) > 0:
                    film.ID = data[0]
                    film.titre = data[1]
                    film.acteurs.append(data[2])
                    film.acteurs.append(data[3])
                    film.actrices.append(data[4])
                    film.actrices.append(data[5])
                    film.realisateur = data[6]
                    film.noteacteur1 = float(data[9])
                    film.noteacteur2 = float(data[10])
                    film.noteactrice1 = float(data[11])
                    film.noteactrice2 = float(data[12])
                    film.noterealisateur = float(data[13])
                    # film.type = data[8]
                    # print(film.realisateur)
                    Test.append(film)
                line = fp.readline()
        nav = naive_bayes(Datasetfilms, Test)
        # print(nav)
        self.message["text"] = nav

    def fonction2(self):
        Datasetfilms = []
        with open("TrainDATA.txt") as fp:
            line = fp.readline()
            while line:
                # print("o")
                film = Film()
                data = line.split(';')
                if len(data) > 0:
                    film.ID = data[0]
                    film.titre = data[1]
                    film.acteurs.append(data[2])
                    film.acteurs.append(data[3])
                    film.actrices.append(data[4])
                    film.actrices.append(data[5])
                    film.realisateur = data[6]
                    film.noteacteur1 = float(data[9])
                    film.noteacteur2 = float(data[10])
                    film.noteactrice1 = float(data[11])
                    film.noteactrice2 = float(data[12])
                    film.noterealisateur = float(data[13])
                    film.type = data[8]
                    Datasetfilms.append(film)
                line = fp.readline()
        Test = []
        with open("TestDATA.txt") as fp:
            line = fp.readline()
            while line:
                film = Film()
                data = line.split(';')
                if len(data) > 0:
                    film.ID = data[0]
                    film.titre = data[1]
                    film.acteurs.append(data[2])
                    film.acteurs.append(data[3])
                    film.actrices.append(data[4])
                    film.actrices.append(data[5])
                    film.realisateur = data[6]
                    film.noteacteur1 = float(data[9])
                    film.noteacteur2 = float(data[10])
                    film.noteactrice1 = float(data[11])
                    film.noteactrice2 = float(data[12])
                    film.noterealisateur = float(data[13])
                    # film.type = data[8]
                    # print(film.realisateur)
                    Test.append(film)
                line = fp.readline()
        self.message["text"] = KNN(5,Test,Datasetfilms)

    
    def fonction3(self):  
        self.message["text"] =deeplearning()

    def fonction4(self):  
        self.message["text"] = "Vous avez cliqué sur la fonction 4"

    def fonction5(self):
        self.message["text"] =NB2.predict()


#Creation de la fenetre
root = Tk()
root.geometry("800x600")
##root.resizable(False, False)
interface = Interface(root)
interface.mainloop()
interface.destroy()