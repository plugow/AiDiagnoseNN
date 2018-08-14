from sklearn.externals import joblib
import sys
arg = sys.argv[1].split(',')
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
result = loaded_model.predict([arg])
print(result)