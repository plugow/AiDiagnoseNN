import random
import numpy as np
from itertools import chain, repeat, islice
import io
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsClassifier

path = 'dictionary.txt'
encoding = 'utf-8'
symptom_index_map = {}
current_index = 1
max_vector_length = 5


# load dictionary
with io.open(path, 'r', encoding=encoding) as f:
    for line in f:
        symptom_index_map[line.strip("\n")] = current_index
        current_index += 1


def symptoms_to_ids(path_to_symptoms, vocabulary):
    with io.open(path_to_symptoms, 'r', encoding=encoding) as f:
        symptoms = []
        for line in f:
            symptoms.append(line.strip("\n"))
    f.close()
    return [vocabulary.get(s) for s in symptoms]


derma_symptom = symptoms_to_ids('symptoms/dermatolog.txt', symptom_index_map)
gineko_symptom = symptoms_to_ids('symptoms/ginekolog.txt', symptom_index_map)
inter_symptom = symptoms_to_ids('symptoms/internista.txt', symptom_index_map)
kardio_symptom = symptoms_to_ids('symptoms/kardiolog.txt', symptom_index_map)
laryn_symptom = symptoms_to_ids('symptoms/laryngolog.txt', symptom_index_map)
neuro_symptom = symptoms_to_ids('symptoms/neurolog.txt', symptom_index_map)
okul_symptom = symptoms_to_ids('symptoms/okulista.txt', symptom_index_map)
uro_symptom = symptoms_to_ids('symptoms/urolog.txt', symptom_index_map)
wenero_symptom = symptoms_to_ids('symptoms/wenerolog.txt', symptom_index_map)


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


def create_vectors(symptoms, symptom_class):
    vectorSet = set()
    while True:
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 1)), 8, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 2)), 8, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 3)), 8, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 4)), 8, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 5)), 8, 0)))
        # vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 6)), 8, 0)))
        # vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 7)), 8, 0)))
        print(len(vectorSet))
        if len(vectorSet) > 350:
            break
    out = np.array(list(vectorSet))/len(symptom_index_map)
    out[:, -1] = symptom_class
    return out


derma = create_vectors(derma_symptom, 1)
gineko = create_vectors(gineko_symptom, 2)
inter = create_vectors(inter_symptom, 3)
kardio = create_vectors(kardio_symptom, 4)
laryn = create_vectors(laryn_symptom, 5)
neuro = create_vectors(neuro_symptom, 6)
okul = create_vectors(okul_symptom, 7)
uro = create_vectors(uro_symptom, 8)
wenero = create_vectors(wenero_symptom, 9)

data=np.concatenate((derma, gineko, inter, kardio, laryn, neuro, okul, uro, wenero))
np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-40,]
Ytrain = Y[:-40,]
Xtest = X[-40:,]
Ytest = Y[-40:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))

knn = KNeighborsClassifier()
knn.fit(Xtrain, Ytrain)
print("KNN rate:", knn.score(Xtest, Ytest))

filename = 'finalized_model.sav'
joblib.dump(knn, filename)

# for filename in os.listdir(path):
#     inputfile = io.open('symptoms/'+filename, mode='r', encoding=encoding)
#     # for line in inputfile:


# def symptoms_to_vector():
