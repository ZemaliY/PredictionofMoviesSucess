import math
from math import sqrt
from math import pi
from math import exp

class Film:
    def __init__(self):
        self.ID = None
        self.titre = None
        self.realisateur = None
        self.acteurs = []
        self.actrices = []
        self.noteacteur1 = 0
        self.noteactrice1 = 0
        self.noteacteur2 = 0
        self.noteactrice2 = 0
        self.noterealisateur=0
        self.type = None
        self.dist=None

# KNN
def DistEucli(f,dataset):
    for films in dataset :
        distance=float(math.sqrt((f.noteacteur1-films.noteacteur1)*(f.noteacteur1-films.noteacteur1)+
                                 (f.noteactrice1-films.noteactrice1)*(f.noteactrice1-films.noteactrice1)+
                                 (f.noteacteur2 - films.noteacteur2) * (f.noteacteur2 - films.noteacteur2)+
                                 (f.noteactrice2 - films.noteactrice2) * (f.noteactrice2 - films.noteactrice2)
                                 +
                                 (f.noterealisateur - films.noterealisateur) * (f.noterealisateur - films.noterealisateur)
                                 ))
        films.dist=distance
    return dataset

def KNN(k,listeEntr,listeT):
    a='Flop'
    b='Average'
    c='Hit'
    prediction=""
    for i in range(len(listeEntr)):
        c1=0
        c2=0
        c3=0
        list=DistEucli(listeEntr[i],listeT)
        listeT=sorted(list, key=lambda film: film.dist, reverse=False)
        for j in range (k):
            if listeT[j].type=="0":
                c1=c1+1
            if listeT[j].type=="1":
                c2=c2+1
            if listeT[j].type=="2":
                c3=c3+1
        if c1 > c2 and c1 > c3:
            listeEntr[i].type = a
        if c2 > c1 and c2 > c3:
            listeEntr[i].type = b
        if c3 > c2 and c3 > c1:
            listeEntr[i].type = c
        # if c1==c2 and c2>  c3:
        #     listeEntr[i].type = a+'-'+b
    p=0
    for filpred in listeEntr:
        p+=1
        prediction+=filpred.titre+ ' est un '+filpred.type+'\n'
    return (prediction)

# Naive Bayes
def separate_by_class ( dataset ):
    separated = dict()
    for film in dataset:
        class_value = film.type
        if ( class_value not in separated ):
            separated[class_value] = list()
        separated[class_value].append(film)
    return separated
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

def summarize_dataset(dataset):
    noteactrice1=list()
    noteacteur1=list()
    noteactrice2 = list()
    noteacteur2 = list()
    noterealisateur=list()
    for film in dataset:
        noteactrice1.append(film.noteactrice1)
        noteacteur1.append(film.noteacteur1)
        noteactrice2.append(film.noteactrice2)
        noteacteur2.append(film.noteacteur2)
        noterealisateur.append(film.noterealisateur)
    act=list()
    act.append(mean(noteacteur1))
    act.append(stdev(noteacteur1))
    act.append(len(noteacteur1))
    act2 = list()
    act2.append(mean(noteacteur2))
    act2.append(stdev(noteacteur2))
    act2.append(len(noteacteur2))
    bct = list()
    bct.append(mean(noteactrice1))
    bct.append(stdev(noteactrice1))
    bct.append(len(noteactrice1))
    bct2 = list()
    bct2.append(mean(noteactrice2))
    bct2.append(stdev(noteactrice2))
    bct2.append(len(noteactrice2))
    rea = list()
    rea.append(mean(noterealisateur))
    rea.append(stdev(noterealisateur))
    rea.append(len(noterealisateur))
    summaries=[]
    summaries.append(act)
    summaries.append(bct)
    summaries.append(act2)
    summaries.append(bct2)
    summaries.append(rea)
    return summaries
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    return probabilities

def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    listere=list()
    predictions = ""
    for row in test:
        output = predict(summarize, [row.noteacteur1,row.noteacteur2,row.noteactrice1,row.noteactrice2,row.noterealisateur])
        # if output == '2':
        #     output = 'HIT'
        # if output == '1':
        #     output = 'AVERAGE'
        # if output == '0':
        #     output = 'FLOP'
        predictions+=row.titre+' est un : HIT:'+str(output['2'])+', AVERAGE:'+str(output['1'])+', FLOP:'+str(output['0'])+'\n'
    return (predictions)

if __name__=='__main__':
    Datasetfilms=[]
    with open("TrainDATA.txt") as fp:
        line = fp.readline()
        while line:
            # print("o")
            film=Film()
            data = line.split(';')
            if len(data) >0:
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
                film.type =data[8]
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
    # print(Datasetfilms[0].type)
    # separe=separate_by_class(Datasetfilms)
    # # sum=summarize_dataset(Datasetfilms)
    # # print(sum)
    # sumclass=summarize_by_class(Datasetfilms)
    # print (sumclass)
    # # for label in separe:
    # #     print(label)
    # #     for row in separe[label]:
    # #         print(row.titre)
    # # print([separe["0"][1].noteacteur,separe["0"][1].noteactrice])
    # proba=calculate_class_probabilities(sumclass,[separe["2"][1].noteacteur1,separe["2"][1].noteacteur2,separe["2"][1].noteactrice1,separe["2"][1].noteactrice2])
    # print(proba)
    # pred=predict(sumclass,[separe["2"][1].noteacteur1,separe["2"][1].noteacteur2,separe["2"][1].noteactrice1,separe["2"][1].noteactrice2])
    # print(pred)
    nav=naive_bayes(Datasetfilms,Test)
    print(nav)
    # for fil in Datasetfilms:
    #     print(fil.acteurs)
    # print(len(Test))
    # Test=Datasetfilms
    # print(KNN(7,Test,Datasetfilms))




