import csv
import pandas as pd
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# a definir
rules = [("acteur", 292_318, 2_334_348), ("actrice", 292_318, 1_615_654), ("realisateur", 250_000, 1_500_000)]


def Read_film(file, sep=';'):
    # transformation du csv en dataframe
    df = pd.read_csv(file, sep=sep, header=None)
    df = df.drop(0, axis=1)
    # df=df.drop(9,axis=1)
    mat = np.matrix(df)

    return mat


def Read_real(file, sep=';'):
    # transformation du csv en dataframe
    df = pd.read_csv(file, sep=sep, header=None)
    df = df.drop(6, axis=1)
    mat = np.matrix(df)

    return mat


def FilterString(string, upper=False):
    if not string:
        return string
    rules = [('é', 'e'), ('è', 'e'), ('ê', 'e'), ('ë', 'e'), ('ç', 'c'), ('à', 'a'), ('î', 'i'), ('ï', 'i'), ('ô', 'o'),
             ('û', 'u'), ('ü', 'u')]
    tmp = string.lower()
    for x, y in rules:
        tmp = tmp.replace(x, y)
    return tmp


# retourne pour chaque feature la proba d'appartenance a une des 3 classes en fonction du nb dentree de ses films
# retourne aussi les date d1 et d5
# P1
def PaSachb(mat, l_predict, rules, para, d_proba):
    c = 0
    a = -1
    b = -1

    if para == 1:  # acteur
        (a, b) = (0, 1)
    if para == 2:  # actrice
        (a, b) = (2, 3)
    if para == 3:  # real
        a = 4
        c = 1

    for row in mat:
        flop = 0
        avg = 0
        hit = 0
        d1 = 0
        d2 = 0

        if row[0, 0] == l_predict[a]:
            c = c + 1
            for i in range(1, 6):
                if row[0, i] < rules[0][1]:
                    flop = flop + 1
                elif row[0, i] > rules[0][2]:
                    hit = hit + 1
                else:
                    avg = avg + 1

            # d1=row[0,6]
            # d5=row[0,7]

            d_proba.update({row[0, 0]: [flop / 5, avg / 5, hit / 5]})

        if row[0, 0] == l_predict[b] and (para == 1 or para == 2):
            c = c + 1
            for i in range(1, 6):
                if row[0, i] < rules[0][1]:
                    flop = flop + 1
                elif row[0, i] > rules[0][2]:
                    hit = hit + 1
                else:
                    avg = avg + 1

            # d1=row[0,6]
            # d5=row[0,7]

            d_proba.update({row[0, 0]: [flop / 5, avg / 5, hit / 5]})

        if c == 2 or (c == 1 and para == 3):
            break

    for key, val in d_proba.items():
        for i in range(len(val)):
            if d_proba[key][i] == 0:
                d_proba[key][i] = 0.001

    return d_proba


# compte le nombre de films appartenants a chacune des classes dans le training
def Count_training_classes(mat_film):
    flop = 0
    average = 0
    hit = 0
    film = 0
    d_count = {}

    for row in mat_film:
        film = film + 1
        if row[0, 7] == 0:
            flop = flop + 1
        if row[0, 7] == 1:
            average = average + 1
        if row[0, 7] == 2:
            hit = hit + 1

    d_count.update({'nb film': int(film)})
    d_count.update({'flop': int(flop)})
    d_count.update({'avg': int(average)})
    d_count.update({'hit': int(hit)})

    return d_count


# P(B|A)P(A)
def ComputeP1P2(d_count, d_proba):
    d_p1p2 = {}
    p1_flop = 1
    p1_avg = 1
    p1_hit = 1

    for key, value in d_proba.items():
        p1_flop = p1_flop * value[0]
        p1_avg = p1_avg * value[1]
        p1_hit = p1_hit * value[2]

    d_p1p2.update({"p_flop": p1_flop * (d_count['flop'] / d_count['nb film'])})
    d_p1p2.update({"p_avg": p1_avg * (d_count['avg'] / d_count['nb film'])})
    d_p1p2.update({"p_hit": p1_hit * (d_count['hit'] / d_count['nb film'])})

    return d_p1p2


def num_film(file, sep=';'):
    d_nbfilm = {}
    df = pd.read_csv(file, sep=sep, header=None)
    df = df.drop([0, 2, 3, 4, 5, 6], axis=1)
    m = np.matrix(df)
    for item in m:
        d_nbfilm[item[0, 0]] = int(item[0, 1])

    return d_nbfilm


def num_film2(file, sep=';'):
    d_nbfilm = {}
    df = pd.read_csv(file, sep=sep, header=None)
    df = df.drop([1, 2, 3, 4, 5], axis=1)
    m = np.matrix(df)
    for item in m:
        d_nbfilm[item[0, 0]] = int(item[0, 1])

    return d_nbfilm


def train(file, sep=';'):
    # transformation du csv en dataframe
    df = pd.read_csv(file, sep=sep, header=None)
    # df=df.drop([1,2,3,4,5],axis=1)
    mat = np.matrix(df)

    return mat


def ComputeBayes2(l_predict, d_nbfilm, d_p1p2):
    s = 0
    maxi = []
    maxi2 = []
    a = 1

    for value in list(d_nbfilm.values()):
        s = s + value
    for item in l_predict:
        a += d_nbfilm[item]
    a = a / value

    for key, value in d_p1p2.items():
        value = value / a
        if key == 'p_hit':
            classe = "hit"
        elif key == 'p_avg':
            classe = "average"
        elif key == 'p_flop':
            classe = "flop"
        maxi.append(value)

    #         if value>maxi[1]:
    #             maxi=[key,value]

    #     for key,value in d_p1p2.items():
    #         print(key,value)

    return maxi


def predict():
    fic = open("acteur1.csv", "r")
    mat_acteur = Read_film(fic)
    fic.close()

    fic = open("note_actrice.csv", "r")
    mat_actrice = Read_film(fic)
    fic.close()

    fic = open("film1.csv", "r")
    mat_film = Read_film(fic)
    fic.close()

    fic = open("realisateur.csv", "r")
    mat_real = Read_real(fic)
    fic.close()

    fic = open("train.csv", "r")
    mme = train(fic)
    fic.close()
    string = ''

    for row in mme:
        l_predict = [FilterString(row[0, 2]), FilterString(row[0, 3]), row[0, 4], row[0, 5], row[0, 6]]
        d_proba = {}
        d_proba2 = {}
        # l_predict=["leonardo dicaprio","johnny depp","Jennifer Aniston","Natalie Portman","Steven Spielberg"]
        # l_predict=["harrison ford","omar sy","Karen Gillan","Karen Gillan"]
        d_proba = PaSachb(mat_acteur, l_predict, rules, 1, d_proba)
        d_proba = PaSachb(mat_actrice, l_predict, rules, 2, d_proba)
        # d_proba=PaSachb(mat_real,l_predict,rules,3,d_proba)
        for key, val in d_proba.items():
            for i in range(len(val)):
                if d_proba[key][i] == 0:
                    d_proba[key][i] = 0.001

        d_count = Count_training_classes(mat_film)
        d_p1p2 = ComputeP1P2(d_count, d_proba)

        d_nbfilm = num_film(open("acteur1.csv", "r"))
        d_nbfilm2 = num_film(open("actrice.csv", "r"))
        d_nbreal = num_film2(open("realisateur.csv", "r"))
        d_nbfilm.update(d_nbfilm2)
        d_nbfilm.update(d_nbreal)
        maxi = ComputeBayes2(l_predict, d_nbfilm, d_p1p2)
        maxi2 = list(reversed(sorted(maxi)))

        # print(bcolors.BOLD +str(row[0,1])+ bcolors.ENDC+" est un: ",end='')
        string += str(row[0, 1] + ' est un: ')
        c = 0
        for i in maxi2:
            # c=0
            ind = maxi.index(i)
            if ind == 0:
                cat = 'FLOP'
            elif ind == 1:
                cat = 'AVERAGE'
            elif ind == 2:
                cat = 'HIT'
            if c == 0:
                # print(bcolors.OKGREEN +cat+ bcolors.ENDC+":"+str(i)+",", end='')
                string = string + cat + ':' + str(i) + ","
            elif c == 1:
                # print(bcolors.WARNING +cat+ bcolors.ENDC+":"+str(i)+",",end='')
                string = string + cat + ':' + str(i) + ","
            elif c == 2:
                # print(bcolors.FAIL +cat+ bcolors.ENDC+":"+str(i))
                string = string + cat + ':' + str(i) + '\n'
            c += 1

    return string


