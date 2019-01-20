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
    return [vocabulary.get(s) for s in symptoms]


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

mlp = MLPClassifier(max_iter=500, activation='relu', solver='lbfgs')
mlp.fit(Xtrain, Ytrain)
filename = 'finalized_model.sav'
joblib.dump(mlp, filename)
print("MLP rate:", mlp.score(Xtest, Ytest))
loss_values = [2.10522e+00
,1.99285e+00
,1.89471e+00
,1.66665e+00
,1.49445e+00
,1.38607e+00
,1.22553e+00
,1.15103e+00
,1.05023e+00
,9.75009e-01
,9.40433e-01
,9.31047e-01
,9.18802e-01
,8.96668e-01
,8.47320e-01
,8.05345e-01
,7.54906e-01
,6.60718e-01
,6.27426e-01
,6.09714e-01
,5.98887e-01
,5.91009e-01
,5.66102e-01
,5.60035e-01
,5.22165e-01
,5.11396e-01
,4.89892e-01
,4.88684e-01
,4.71424e-01
,4.65640e-01
,4.56954e-01
,4.41962e-01
,4.35892e-01
,4.22858e-01
,4.15659e-01
,4.09410e-01
,4.06776e-01
,3.99112e-01
,3.96670e-01
,3.92952e-01
,3.90066e-01
,3.80691e-01
,3.68882e-01
,3.49163e-01
,3.37240e-01
,3.33572e-01
,3.26579e-01
,3.22721e-01
,3.08152e-01
,2.90898e-01
,2.73699e-01
,2.63185e-01
,2.53204e-01
,2.48219e-01
,2.34651e-01
,2.20981e-01
,2.13437e-01
,2.03993e-01
,2.00915e-01
,1.98660e-01
,1.94946e-01
,1.92206e-01
,1.86784e-01
,1.83773e-01
,1.80209e-01
,1.77287e-01
,1.72301e-01
,1.69275e-01
,1.64346e-01
,1.61459e-01
,1.60705e-01
,1.59416e-01
,1.54683e-01
,1.49665e-01
,1.46694e-01
,1.44863e-01
,1.41041e-01
,1.39486e-01
,1.38472e-01
,1.37627e-01
,1.35769e-01
,1.33714e-01
,1.32776e-01
,1.31594e-01
,1.30748e-01
,1.29719e-01
,1.26885e-01
,1.22932e-01
,1.20811e-01
,1.18162e-01
,1.17564e-01
,1.16113e-01
,1.14888e-01
,1.14430e-01
,1.13580e-01
,1.13221e-01
,1.12953e-01
,1.12350e-01
,1.10637e-01
,1.09922e-01
,1.08323e-01
,1.07504e-01
,1.05933e-01
,1.05042e-01
,1.03590e-01
,1.03025e-01
,1.02627e-01
,1.02210e-01
,1.01412e-01
,1.00498e-01
,9.89228e-02
,9.77023e-02
,9.58897e-02
,9.46127e-02
,9.42017e-02
,9.34900e-02
,9.30430e-02
,9.26026e-02
,9.22977e-02
,9.20653e-02
,9.17616e-02
,9.14223e-02
,9.10489e-02
,9.04803e-02
,8.96435e-02
,8.85999e-02
,8.76297e-02
,8.74332e-02
,8.68181e-02
,8.57799e-02
,8.53043e-02
,8.50758e-02
,8.48772e-02
,8.43459e-02
,8.41099e-02
,8.29732e-02
,8.20237e-02
,8.10760e-02
,8.05780e-02
,7.99667e-02
,7.97647e-02
,7.94274e-02
,7.93541e-02
,7.91452e-02
,7.88279e-02
,7.85458e-02
,7.79546e-02
,7.76641e-02
,7.66703e-02
,7.62471e-02
,7.61561e-02
,7.59170e-02
,7.52447e-02
,7.45886e-02
,7.43003e-02
,7.38103e-02
,7.27219e-02
,7.22313e-02
,7.19534e-02
,7.14945e-02
,7.13702e-02
,7.11821e-02
,7.11401e-02
,7.08638e-02
,7.07615e-02
,7.06619e-02
,7.03669e-02
,6.98670e-02
,6.94938e-02
,6.91549e-02
,6.89585e-02
,6.82836e-02
,6.74894e-02
,6.64460e-02
,6.58652e-02
,6.49216e-02
,6.45420e-02
,6.37818e-02
,6.33898e-02
,6.29685e-02
,6.26837e-02
,6.22393e-02
,6.19123e-02
,6.13969e-02
,6.09622e-02
,6.05662e-02
,6.02701e-02
,5.98627e-02
,5.85496e-02
,5.76626e-02
,5.60950e-02
,5.53618e-02
,5.44272e-02
,5.41291e-02
,5.39565e-02
,5.35137e-02
,5.30872e-02
,5.25854e-02
,5.24002e-02
,5.22279e-02
,5.18412e-02
,5.11989e-02
,5.08096e-02
,5.02249e-02
,4.98349e-02
,4.92615e-02
,4.87415e-02
,4.85094e-02
,4.82583e-02
,4.79298e-02
,4.78543e-02
,4.76584e-02
,4.67779e-02
,4.61670e-02
,4.57595e-02
,4.56744e-02
,4.55620e-02
,4.50700e-02
,4.47102e-02
,4.44817e-02
,4.42141e-02
,4.40360e-02
,4.39191e-02
,4.37731e-02
,4.35294e-02
,4.31905e-02
,4.29666e-02
,4.25166e-02
,4.24357e-02
,4.21911e-02
,4.17561e-02
,4.15419e-02
,4.12935e-02
,4.11489e-02
,4.09632e-02
,4.08602e-02
,4.07186e-02
,4.05000e-02
,4.03830e-02
,4.00121e-02
,4.00020e-02
,3.95862e-02
,3.94659e-02
,3.93622e-02
,3.93122e-02
,3.91786e-02
,3.88580e-02
,3.87001e-02
,3.86162e-02
,3.85758e-02
,3.84271e-02
,3.79354e-02
,3.77015e-02
,3.76085e-02
,3.73836e-02
,3.69958e-02
,3.67393e-02
,3.66475e-02
,3.65593e-02
,3.65297e-02
,3.64131e-02
,3.63520e-02
,3.62204e-02
,3.61618e-02
,3.60849e-02
,3.59396e-02
,3.57970e-02
,3.56439e-02
,3.54560e-02
,3.52515e-02
,3.51978e-02
,3.48464e-02
,3.46755e-02
,3.46386e-02
,3.45332e-02
,3.45098e-02
,3.44553e-02
,3.42794e-02
,3.40550e-02
,3.39319e-02
,3.37843e-02
,3.36457e-02
,3.35399e-02
,3.33905e-02
,3.32176e-02
,3.31664e-02
,3.31411e-02
,3.30874e-02
,3.29214e-02
,3.26861e-02
,3.26421e-02
,3.25858e-02
,3.25276e-02
,3.24947e-02
,3.23749e-02
,3.21855e-02
,3.20367e-02
,3.19457e-02
,3.19030e-02
,3.18832e-02
,3.18365e-02
,3.16913e-02
,3.16502e-02
,3.15822e-02
,3.15313e-02
,3.15080e-02
,3.14906e-02
,3.14688e-02
,3.14421e-02
,3.14167e-02
,3.13762e-02
,3.12861e-02
,3.11567e-02
,3.10338e-02
,3.09358e-02
,3.08262e-02
,3.07850e-02
,3.07289e-02
,3.05792e-02
,3.04594e-02
,3.03604e-02
,3.03366e-02
,3.02886e-02
,3.01432e-02
,3.00815e-02
,2.99882e-02
,2.99597e-02
,2.99314e-02
,2.99075e-02
,2.98862e-02
,2.98608e-02
,2.98454e-02
,2.98010e-02
,2.97678e-02
,2.97354e-02
,2.96686e-02
,2.95866e-02
,2.94805e-02
,2.94222e-02
,2.93455e-02
,2.92772e-02
,2.92622e-02
,2.92300e-02
,2.91932e-02
,2.91454e-02
,2.89655e-02
,2.88925e-02
,2.88006e-02
,2.87171e-02
,2.85566e-02
,2.85116e-02
,2.84538e-02
,2.84067e-02
,2.83817e-02
,2.83467e-02
,2.83247e-02
,2.82620e-02
,2.81660e-02
,2.80693e-02
,2.79612e-02
,2.78331e-02
,2.77826e-02
,2.77133e-02
,2.76719e-02
,2.76326e-02
,2.76015e-02
,2.75814e-02
,2.74519e-02
,2.74034e-02
,2.73542e-02
,2.72943e-02
,2.71927e-02
,2.71705e-02
,2.70789e-02
,2.70556e-02
,2.70286e-02
,2.69956e-02
,2.69559e-02
,2.68767e-02
,2.67578e-02
,2.66089e-02
,2.65226e-02
,2.63195e-02
,2.62154e-02
,2.62092e-02
,2.61227e-02
,2.60885e-02
,2.60419e-02
,2.59681e-02
,2.58776e-02
,2.57571e-02
,2.56574e-02
,2.56147e-02
,2.55190e-02
,2.54774e-02
,2.53405e-02
,2.53047e-02
,2.52939e-02
,2.52122e-02
,2.51663e-02
,2.50895e-02
,2.49755e-02
,2.48397e-02
,2.46398e-02
,2.45220e-02
,2.44129e-02
,2.43354e-02
,2.42653e-02
,2.41373e-02
,2.40558e-02
,2.40349e-02
,2.39836e-02
,2.38450e-02
,2.37795e-02
,2.36277e-02
,2.35615e-02
,2.34124e-02
,2.33212e-02
,2.31169e-02
,2.29868e-02
,2.29359e-02
,2.28704e-02
,2.28141e-02
,2.27195e-02
,2.24609e-02
,2.22990e-02
,2.21434e-02
,2.20566e-02
,2.19838e-02
,2.19083e-02
,2.16885e-02
,2.15906e-02
,2.14145e-02
,2.12628e-02
,2.11629e-02
,2.09949e-02
,2.09442e-02
,2.08489e-02
,2.06753e-02
,2.06035e-02
,2.04835e-02
,2.03945e-02
,2.03254e-02
,2.02418e-02
,2.01857e-02
,2.01130e-02
,1.99443e-02
,1.98229e-02
,1.95705e-02
,1.94426e-02
,1.93097e-02]

# plt.plot(loss_values)
# plt.title ="Wykres błędu dla epoki"
# plt.xlabel='Epoka'
# plt.ylabel='Wartość błędu'
# plt.show()




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# plot_learning_curve(mlp, 'cos', Xtrain, Ytrain, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
