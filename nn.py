import random
import numpy as np
from itertools import chain, repeat, islice
import io
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib

path = 'dictionary.txt'
pathMap = 'map.txt'
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
    return [vocabulary.get(s)/len(symptom_index_map) for s in symptoms]


derma_symptom = symptoms_to_ids('symptoms/dermatolog.txt', symptom_index_map)
gineko_symptom = symptoms_to_ids('symptoms/ginekolog.txt', symptom_index_map)
inter_symptom = symptoms_to_ids('symptoms/internista.txt', symptom_index_map)
kardio_symptom = symptoms_to_ids('symptoms/kardiolog.txt', symptom_index_map)
laryn_symptom = symptoms_to_ids('symptoms/laryngolog.txt', symptom_index_map)
neuro_symptom = symptoms_to_ids('symptoms/neurolog.txt', symptom_index_map)
okul_symptom = symptoms_to_ids('symptoms/okulista.txt', symptom_index_map)
uro_symptom = symptoms_to_ids('symptoms/urolog.txt', symptom_index_map)
# wenero_symptom = symptoms_to_ids('symptoms/wenerolog.txt', symptom_index_map)


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


def create_vectors(symptoms, symptom_class):
    vectorSet = set()
    while True:
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 1)), 6, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 2)), 6, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 3)), 6, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 4)), 6, 0)))
        vectorSet.add(tuple(pad(sorted(random.sample(symptoms, 5)), 6, 0)))
        if len(vectorSet) > 350:
            break
    out_not_normalized=np.array(list(vectorSet))
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
# wenero = create_vectors(wenero_symptom, 9)

data=np.concatenate((derma, gineko, inter, kardio, laryn, neuro, okul, uro))
np.random.shuffle(data)
data=data[:3160,]
X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# model = LogisticRegression()
# model.fit(Xtrain, Ytrain)
# print("Classification rate:", model.score(Xtest, Ytest))
#(max_iter=2500, activation='tanh', solver='lbfgs', tol=1e-12, verbose=True, hidden_layer_sizes=(100,100) ) 0.88

mlp = MLPClassifier(max_iter=2500, activation='tanh', solver='lbfgs', tol=1e-12, verbose=False, hidden_layer_sizes=(100,100) )
mlp.fit(Xtrain, Ytrain)
filename = 'finalized_model.sav'
joblib.dump(mlp, filename)
print("MLP rate:", mlp.score(Xtest, Ytest))

