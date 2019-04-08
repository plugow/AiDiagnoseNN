from sklearn.externals import joblib
import sys
# arg = sys.argv[1].split(',')
arg = [0.5277777777777778, 0.5462962962962963, 0, 0, 0]
filename = 'finalized_model_best.sav'
loaded_model = joblib.load(filename)
conf = loaded_model.predict_proba([arg])
result = loaded_model.predict([arg])
print(result)
print(int(result))
print(conf[0][int(result)-1])
print(conf[0])