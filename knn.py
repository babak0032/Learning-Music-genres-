import scipy.io
import os
import math
import numpy

data = scipy.io.loadmat("gtzan.mat")

#print data['fold9_features'][0]

def ecludianDistance(list1, list2):  #funtion that calculates the Ecludian distance between two points of n-dimesnsion.
	distance = 0
	for i in range(len(list2)):
		distance = distance + (list1[i] - list2[i])**2
	return	distance    # this should actually return sqrt(distance), but since we only wish to compare distanes, we don't need to sqrt because it will just extra calculatuin. for example if 2 < 5 then it mean 4 < 25 as well. 

#test:
#print ecludianDistance(data['fold1_features'][0], data['fold2_classes'][0])

def kNearest(k, song, matrix): #this function takes a song and a fold_features(say fold8-features) and ofcourse of the 'K' number, and returns the 'K' shortest points(songs) in fold8-feature to the argument song and their index in fold8-features. 
	dic = {}
	for i in range(len(matrix)):
		index = i
		dic[ecludianDistance(song, matrix[i])] = (index)  #a dictionary of all songs distanes and their index

	distances = dic.keys()
	distances.sort()

	kShortestDistances = []
	for x in range(k):
		kShortestDistances.append((distances[x], dic[distances[x]])) #after sorting keys, pick the first 'K' ones and their indexes.
		del dic[distances[x]]
		
	return kShortestDistances	

#test:
#print kNearest(1, data['fold1_features'][0], data['fold8_features'])


#Here I had to records the name of all fetures and classes for definig method purposes.
folds = ['fold1_features', 'fold2_features', 'fold3_features', 'fold4_features', 'fold5_features', 'fold6_features', 'fold7_features', 'fold8_features', 'fold9_features', 'fold10_features']	
classes = ['fold1_classes', 'fold2_classes', 'fold3_classes', 'fold4_classes', 'fold5_classes', 'fold6_classes', 'fold7_classes', 'fold8_classes', 'fold9_classes', 'fold10_classes']

def choose_id(dictionary):  #this is method for classifying a song given its 'K' nearest closest song ids and their distances. It will pick the id that acuurs in the dictionary the most.
	genres = []
	for i in range(1, 11):
		genre_counter = 0
		for value in dictionary.values():
			if (value == i):
				genre_counter += 1
		genres.append(genre_counter)	
	
	maxNumber = max(genres)      
	for genre in genres:        
		if (maxNumber == genre):       #if there is a tie:
			if (genres.index(maxNumber) != genres.index(genre)):  #and tie is between two different genre:
				del dictionary[dictionary.keys()[-1]]             #delete the furthest point
				choose_id(dictionary)	               			  #recurse

	return genres.index(maxNumber)	+ 1	


								  #'classify' will take a song and the 'K' and the name of its feature_fold, then it will return the id of genre that the song has benn classifed whih is a number from 1 to 10.			
def classify(k, song, foldName):  #foldName is the name of feature fold of the TEST SONG, not the traning ones!
	closestsSongs = {}
	for fold in folds:            #from all fold_feqtures, pick the 'k' closest songs from each and put it in a ditionary with their fold-name and their index within their fold. So this dictionary will have a length of k*n.
		if (fold != foldName):
			points = kNearest(k, song, data[fold])
			for p in points:
				closestsSongs[p[0]] = [p[1], fold]
	
	closestDistances = closestsSongs.keys()     
	closestDistances.sort()

	KClosestSongs = {}
	for x in range(k):        #Now pick the 'K' closest songs out of those songs(the dictionary above with the length of k*n).
		KClosestSongs[closestDistances[x]] = closestsSongs[closestDistances[x]]
		del closestsSongs[closestDistances[x]] 	

	KClosestSongsIDs = {}  #Now pick the ID of the K point chosen and put them in a dictionary with their distances.
	for d in KClosestSongs.keys():
		classNumber = folds.index(KClosestSongs[d][1])
		KClosestSongsIDs[d] = data['fold' + str(classNumber + 1) + '_classes'][KClosestSongs[d][0]][0]

	return choose_id(KClosestSongsIDs) 

#test:
#print classify(1, data['fold1_features'][0], 'fold1_features')
#print classify(3, data['fold1_features'][0], 'fold1_features')
#print classify(5, data['fold1_features'][0], 'fold1_features')


def totalClassification(k, matrixSong): #this method will do classifiation for a complete feature-fold, and it returns the ids in a list.
	id_classes = []
	for i in data[matrixSong]:
		id_classes.append(classify(k, i, matrixSong))
	return id_classes

#test:
#print totalClassification(1, 'fold1_features')

def accuracy(ids1, ids2):
	actualIDs = [item for sublist in ids2 for item in sublist]
	#print actualIDs

	correctClassificationNumber = 0
	for i in range(len(actualIDs)):
		if (ids1[i] == actualIDs[i]):
			correctClassificationNumber += 1
	return (correctClassificationNumber/float(len(ids1)))

#test:
#test1 = totalClassification(1, 'fold9_features')
#print accuracy(test1, data['fold9_classes'])	

def totalAccuracy(k):
	sum_average = 0.0
	for	i in range(10):
		print accuracy(totalClassification(k, folds[i]), data[classes[i]])
		sum_average += accuracy(totalClassification(k, folds[i]), data[classes[i]])/10.0
		#print sum_average
	return	sum_average

#test:
#print totalAccuracy(1)
#print totalAccuracy(3)
#print totalAccuracy(5)

#OneOther way to calculate accuracy is using confusion matrix. I implement the method above for calculating accuracy only because I find it easier to understand.
def confusion(ids1, ids2):
	confusionMatrix = numpy.zeros((10,10), dtype=int)
	actualIDs = [item for sublist in ids2 for item in sublist]
	for count1, count2 in zip(ids1, actualIDs):
		confusionMatrix[count1-1][count2-1] += 1
	return confusionMatrix	
	
#print confusion(test1, data['fold9_classes'])	

def fullConfusion(k):
	fullConfusionMatrix = numpy.zeros((10,10), dtype=int)
	for fold in folds:
		ids = totalClassification(k, fold)
		fullConfusionMatrix += confusion(ids, data['fold' + str(folds.index(fold) + 1) + '_classes'])
	print fullConfusionMatrix
	
	accuracy = 0
	for i in range(10):
		#print fullConfusionMatrix[i][i]
		#print float(sum(fullConfusionMatrix[i]))
		accuracy += (fullConfusionMatrix[i][i])/float(sum(fullConfusionMatrix[i]))

	return accuracy/10	

#print fullConfusion(1)
#print fullConfusion(5)
					
				