import scipy.io
import os
import math
import numpy 
import knn



#os.chdir("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2")
data = scipy.io.loadmat("gtzan.mat")



genres = ['genre1', 'genre2','genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10'] 
folds = ['fold1_features', 'fold2_features', 'fold3_features', 'fold4_features', 'fold5_features', 'fold6_features', 'fold7_features', 'fold8_features', 'fold9_features', 'fold10_features']	
classes = ['fold1_classes', 'fold2_classes', 'fold3_classes', 'fold4_classes', 'fold5_classes', 'fold6_classes', 'fold7_classes', 'fold8_classes', 'fold9_classes', 'fold10_classes']

								#this method will list all songs in each id for every feature-fold except the fold given(which is the test fold). it returns a dictionary with id of each genre and its points in all folds.
def classesFinder(FoldName):
	genresSongs = {'genre1' : [], 'genre2': [],'genre3': [], 'genre4': [], 'genre5': [], 'genre6': [], 'genre7': [], 'genre8': [], 'genre9': [], 'genre10': []}
	for genre in genres:
		genreId = genres.index(genre) + 1
		#print genreId
		for fold in folds:
			foldIndex = folds.index(fold) + 1
			#print foldIndex
			if (fold != FoldName):
				songIndex = 0
				for song in data[fold]:
					if (data['fold' + str(foldIndex) + '_classes'][songIndex] == genreId):
						genresSongs[genre].append(song)
					songIndex += 1
	return genresSongs					 

#test:
#print classesFinder('fold1_features')['genre1']	

def mean(songs):      #finding the mean
	return numpy.mean(songs, axis=0)

#test:
#print mean(classesFinder('fold1_features')['genre1'])


def covarience(songs):      #finding the covariene matrix
	matrix = numpy.array(songs).T
	matrix = numpy.cov(matrix)
	return numpy.diag(numpy.diag(matrix))

#test:
#print covarience(classesFinder('fold1_features')['genre1'])	

def gaussianPosteriorProbability(song, logOfDeterminantOfCovarienceOfTrainigSongs, inverseOfCovarienceOfTrainigSongs, meanOfTrainigSongs): #finding the log posterior probability of a song, given a training set and a class	
	d = numpy.array(song) - numpy.array(meanOfTrainigSongs)
	c = numpy.dot(d, inverseOfCovarienceOfTrainigSongs)
	c = numpy.dot( c,d.T)
	c = c*(-0.5)
	c = c - ((0.5)*(logOfDeterminantOfCovarienceOfTrainigSongs))
	return c

def guassianClassifier(song, FoldName, trainingSongs): #classifying a song (given its foldName), by finding the "log Posterior Probability" for all classes
	#trainingSongs = classesFinder(FoldName)
	probability_Of_Each_Genre = []
	for genre in genres:
		meanOfTrainigSongs = mean(trainingSongs[genre])
		cov = covarience(trainingSongs[genre])
		inverseOfCovarienceOfTrainigSongs = numpy.linalg.inv(cov)
		logOfDeterminantOfCovarienceOfTrainigSongs = numpy.log(numpy.linalg.det(cov))
		probability_Of_Each_Genre.append(gaussianPosteriorProbability(song, logOfDeterminantOfCovarienceOfTrainigSongs, inverseOfCovarienceOfTrainigSongs, meanOfTrainigSongs))
	
	#print probability_Of_Each_Genre
	bestClass = max(probability_Of_Each_Genre)
	return probability_Of_Each_Genre.index(bestClass) + 1

#test:
#testSong = data['fold1_features'][0]
#print guassianClassifier(testSong, 'fold1_features')

def totalGaussianClassification(matrixSongs): #this method will do classifiation for a complete feature-fold, and it returns the ids in a list.
	trainingSongs = classesFinder(matrixSongs)
	id_classes = []
	for song in data[matrixSongs]:
		id_classes.append(guassianClassifier(song, matrixSongs, trainingSongs))
	return id_classes
	
#test:
#print totalGaussianClassification('fold1_features')

def totalAcuuracyGaussian1(): 
	sum_average = 0.0
	for	i in range(10):
		sum_average += knn.accuracy(totalGaussianClassification(folds[i]), data[classes[i]])/10.0
	return	sum_average

#test:
#print totalAcuuracyGaussian1()

def non_digonal_covarience(songs):	#get the sharing covaarience Matrix.
	return numpy.cov((songs), rowvar=0)

def guassianDiscriminantClassifier(song, FoldName, inverseOfCovarienceOfTrainigSongs, logOfDeterminantOfCovarienceOfTrainigSongs, trainingSongs): #classifying a song (given its foldName), by finding the "log Posterior Probability" for all classes
	probability_Of_Each_Genre = []
	for genre in genres:
		meanOfTrainigSongs = mean(trainingSongs[genre])
		probability_Of_Each_Genre.append(gaussianPosteriorProbability(song, logOfDeterminantOfCovarienceOfTrainigSongs, inverseOfCovarienceOfTrainigSongs, meanOfTrainigSongs))
	bestClass = max(probability_Of_Each_Genre)
	return probability_Of_Each_Genre.index(bestClass) + 1

#test:
#testSong = data['fold1_features'][0]
#print guassianDiscriminantClassifier(testSong, 'fold1_features')

def totalGaussianClassificationDiscriminant(matrixSongs): #this method will do classifiation for a complete feature-fold, and it returns the ids in a list.
	trainingSongs = classesFinder(matrixSongs)
	id_classes = []
	allPoints = []
	for fold in folds:
		allPoints.extend(data[fold])
	cov = non_digonal_covarience(allPoints)
	inverseOfCovarienceOfTrainigSongs = numpy.linalg.inv(cov)
	logOfDeterminantOfCovarienceOfTrainigSongs = numpy.log(numpy.linalg.det(cov))		
	for song in data[matrixSongs]:
		id_classes.append(guassianDiscriminantClassifier(song, matrixSongs, inverseOfCovarienceOfTrainigSongs, logOfDeterminantOfCovarienceOfTrainigSongs, trainingSongs))
	return id_classes

#test:
#print totalGaussianClassificationDiscriminant('fold2_features')
#print data['fold2_classes']

def totalAcuuracyGuassian2():
	sum_average = 0.0
	for	i in range(10):
		sum_average += knn.accuracy(totalGaussianClassificationDiscriminant(folds[i]), data[classes[i]])/10.0
	return	sum_average

print totalAcuuracyGuassian2()
