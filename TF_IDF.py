import numpy
import math
import pandas as pd 
#Lectura de archivo limpio y de texto original
archivo = open('Noticias_2P.txt', 'r', encoding='utf-8')
sinStop = open('outNoStop.txt', 'r', encoding='utf-8')
aux=archivo.readlines()
OriginalLines=[]
for item in aux:
    OriginalLines.append(item.split('$')[1])
LineRef=sinStop.readlines()

#Vocabulario
vocabulario = set()
for line in LineRef:
    linevocab=line.split()
    for word in linevocab:
            vocabulario.add(word)

#diccionario
moc = {}
N = len(OriginalLines)
for word in vocabulario:
    li = [0 for i in range(N)]
    li2 = [0 for i in range(N)]
    li3 = [0, 0]
    li4 = [0 for i in range(N)]
    moc.setdefault(word, [li, li2, li3, li4])
numdoc = 0

#frecuencia
for doc in LineRef:
    words = doc.split()
    for word in words:
        if word in moc:
            moc[word][0][numdoc] += 1
    numdoc += 1

#TF
for item in moc:
    for i in range(N):
        if moc[item][0][i] != 0:
            moc[item][1][i] = 1 + math.log(moc[item][0][i], 2)
        else:
            moc[item][1][i] = 0

#n_i
for word in vocabulario:
    for doc in LineRef:
        words = doc.split()
        if word in words:
            moc[word][2][0] += 1

#IDF
for item in moc:
    moc[item][2][1] = math.log(N/moc[item][2][0], 10)

#TF/IDF
for item in moc:
    for i in range(N):
        moc[item][3][i] = moc[item][1][i]*moc[item][2][1]

# Volcado a archivo
cols=[item for item in moc]
rows=[]
for item in moc:
    row=[ num for num in moc[item][3]]
    rows.append(row)
arr = numpy.asarray(rows)
DF = pd.DataFrame(arr, columns=range(1,54), index=cols)
DF=DF.transpose()
DF.to_csv("TF_IDF_TABLE.csv", encoding='cp1252')

#Consulta
consulta = input("Ingrese consulta: ").lower()
consulta = consulta.split()
listadocs = {}
for word in consulta:
    if word in moc:
        i = 0
        for doc in moc[word][3]:
            if doc > 0:
                if i in listadocs:
                    listadocs[i].append([word, doc])
                else:
                    listadocs.setdefault(i, [[word, doc]])
            i += 1
relevant=[]
index=0
for doc in listadocs:
    aux=0.0
    for word in listadocs[doc]:
        aux+=word[1]
    relevant.append([aux, doc])
relevant.sort(reverse=True)

print('\nDOCUMENTOS RELEVANTES\n')
for item in relevant:
    print("Documento {}:  {}".format(item[1]+1, listadocs[item[1]]))
    print(OriginalLines[item[1]])
