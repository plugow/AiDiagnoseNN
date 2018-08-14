# -*- coding: utf-8 -*-

import io
import os
dictionary = io.open('dictionary.txt', mode='w+', encoding="utf-8")

index = 1
dict_index=1
out_text = ''
encoding = 'utf-8'
symptomList = []
path = 'symptoms'

with io.open('dictionary.txt', 'w', encoding=encoding) as f:
    for filename in os.listdir(path):
        print(filename)
        inputfile = io.open('symptoms/'+filename, mode='r', encoding="utf-8")
        for line in inputfile:
            tempLine = line.strip("\n")
            if tempLine not in symptomList:
                symptomList.append(tempLine)
                out_text = "{ name: '" + tempLine+"' },\n"
                f.write(out_text)
                dict_index += 1
            else:
                print(line)
f.close()




