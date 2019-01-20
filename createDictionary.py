# -*- coding: utf-8 -*-

import io
import os
dictionary = io.open('dictionary.txt', mode='w+', encoding="utf-8")
map = io.open('dictionary.txt', mode='w+', encoding="utf-8")

index = 1
dict_index=1
out_text = ''
encoding = 'utf-8'
symptomList = []
path = 'symptoms'
specialization=0

# with io.open('map.txt', 'w', encoding=encoding) as f:
#     for filename in os.listdir(path):
#         print(filename)
#         inputfile = io.open('symptoms/'+filename, mode='r', encoding="utf-8")
#         for line in inputfile:
#             tempLine = line.strip("\n")
#             if tempLine not in symptomList:
#                 symptomList.append(tempLine)
#                 mapValue = str(float(dict_index/108))
#                 out_text = tempLine + "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t" + str(specialization) + "\t\t\t\t\t" + mapValue + "\n"
#                 f.write(out_text)
#                 dict_index += 1
#         specialization = specialization+1
# f.close()

with io.open('dictionary.txt', 'w', encoding=encoding) as f:
    for filename in os.listdir(path):
        print(filename)
        inputfile = io.open('symptoms/'+filename, mode='r', encoding="utf-8")
        for line in inputfile:
            tempLine = line.strip("\n")
            if tempLine not in symptomList:
                symptomList.append(tempLine)
                out_text = tempLine+"\n"
                f.write(out_text)
                dict_index += 1
f.close()


# derma_symptom = []
# gineko_symptom = []
# inter_symptom = []
# kardio_symptom = []
# laryn_symptom = []
# neuro_symptom = []
# okul_symptom = []
# uro_symptom = []




