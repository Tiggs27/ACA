import os
import numpy as np
import pandas as pd
import skimage as sk


dir = "C:/Gustavo Luizon/Cursos/Data Science/Mestrado UC/Disciplinas/Aprendizagem Computacional Avancada/Projeto/desenvolvimento/datasets/dataset_train"

dataset=[]
classification=[]
for subdir in os.listdir(dir):
    print(subdir)
    current_dir = os.path.join(dir, subdir)
    for file in os.listdir(current_dir):
        if subdir == "fake":
            classification.append(0)
        elif subdir == "real":
            classification.append(1)
        else:
            classification.append(-1) 

        if file[-3:] in {'jpg', 'png'}:
            path = os.path.join(current_dir, file)
            img = sk.io.imread(path)
            lista=[]
            for x in range (img.shape[0]):
                for y in range(img.shape[1]):
                    lista.append(img[x][y])                    
            dataset.append(lista)
            print(lista)

dataset=np.array(dataset,dtype=int)
classification = np.array(classification,dtype=int)

csv=pd.DataFrame(np.array(dataset,dtype=str))
csv.to_csv("C:/Gustavo Luizon/Cursos/Data Science/Mestrado UC/Disciplinas/Aprendizagem Computacional Avancada/Projeto/desenvolvimento/datasets/save/dataset.csv")
np.savez('C:/Gustavo Luizon/Cursos/Data Science/Mestrado UC/Disciplinas/Aprendizagem Computacional Avancada/Projeto/desenvolvimento/datasets/save/dataset.npz',dataset=dataset, classification = classification)


#TEST_DATASET READING-------------------------------------------------
#Dataset a ser utilizado na competicao
dir = "C:/Gustavo Luizon/Cursos/Data Science/Mestrado UC/Disciplinas/Aprendizagem Computacional Avancada/Projeto/desenvolvimento/datasets/dataset_test/images"
test_dataset=[]
for file in os.listdir(dir):
    if file[-3:] in {'jpg', 'png'}:
        path = os.path.join(dir, file)
        img = sk.io.imread(path)
        lista=[]
        for x in range (img.shape[0]):
            for y in range(img.shape[1]):
                lista.append(img[x][y])
        test_dataset.append(lista)
        print(lista)

test_dataset=np.array(test_dataset,dtype=int)
np.savez('C:/Gustavo Luizon/Cursos/Data Science/Mestrado UC/Disciplinas/Aprendizagem Computacional Avancada/Projeto/desenvolvimento/datasets/save/test_dataset.npz',test_dataset=test_dataset)

#FIM-------------------------------------------------------

dataset.shape

