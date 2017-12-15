from collections import Counter

def getData(file):
	l = []
        for line in file:
                data = line.split(',')
                genus = data[0]
                genus = genus.split()[0]
                for i in xrange(1,len(data)):
                        data[i] = float(data[i])
                if genus == "Quercus":
                        l.append([data[1:],[1,0,0,0]])
                elif genus == "Acer":
                        l.append([data[1:],[0,1,0,0]])
                elif genus == "Alnus":
                        l.append([data[1:],[0,0,1,0]])
                elif genus == "Populus":
                        l.append([data[1:],[0,0,0,1]])
	return l

def createDataset():
	with open('data_Mar_64.txt') as file:
		a = getData(file)
	
	with open('data_Sha_64.txt') as file:
        	b = getData(file)

	with open('data_Tex_64.txt') as file:
        	c = getData(file)

	dataset = []
	for mar, sha, tex in zip(a,b,c):
		dataset.append([mar[0], sha[0], tex[0], mar[1]])	

	return dataset
