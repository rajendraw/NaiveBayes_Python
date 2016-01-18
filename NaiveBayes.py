__author__ = 'Rajendra'

# Problem statement : Given a sample document (here it's a movie review) we have to
# identify to which class it belongs. Here there are only two classes positive
# and negative. But in general there can be any number of classes.
# We are given a repository of documents already classified as  positive or negative.

# In this problem, we are not classifying an unknown file. Instead we are given 1000 pos 
# and 1000 neg files. We are testing the accuracy of our Naive-Bayes classifier. We combine
# both pos and neg files and split
# the files randomly in the ratio of 2:1, training Vs testing respectively. We train the
# model using training data and test its accuracy on the testing for both sets of
# positive and negative files.

# Problem solution:

# The idea here is, for a given class :
# For all words in all documents : make a dictionary of (word : count of the word)
# find the probability of the word in the class using the formula : (word count)/( All words count)
# But it has to be normalized to account for unknown words by : (word count + 1)/(All words count + Num unique words + 1)
# As these values are very small, take logs of this value so that we avoid floating point errors.
# Once this word probabilities are made for each class, then for a given document 'd', we have to find that class
# for which P(c|d) is maximum and classify 'd' as belonging to that class. Use Bayes's formula to calculate P(c|d)
# P(c|d) = P(c) * P(d|c) / P(d).
# P(d|c) = P( w1,w2,..wn | c) = P(w1|c)*P(w2|c)*.....*P(wn|c) , where w1, w2.. are words in the document 'd'.


import glob as glob
import os as os
import random
import re as re
import argparse
import collections as coll
import math as math

global uniqueWordsSet

uniqueWordsSet = set()

stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
	     'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
	     'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
	     'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
	     'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
	     'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
	     'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
	     'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
	     'own', 'rather','re','s', 'said', 'say', 'says', 'she', 'should', 'since',
	     'so', 'some', 't','than', 'that', 'the', 'their', 'them', 'then',
	     'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
	     've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
	     'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
	     'you', 'your']

################################################################################################

# This function returns the list of all non-empty files in the given directory.
def filelist(path):

    files = glob.glob(path)
    nonEmptyFiles = []
    for f in files:
        if( os.stat(f).st_size != 0):
            nonEmptyFiles.append(f)

    return nonEmptyFiles

################################################################################################

# This function returns the name of the directory specified after '-d' in the command line
def parseArgument():

    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

################################################################################################

# This function splits the list of files into 3 parts.
def getTrainingAndTestingFiles( allFiles ):

    Npos = len(allFiles)
    filesPart1 = allFiles[0 : Npos/3]
    filesPart2 = allFiles[Npos/3 : 2*Npos/3]
    filesPart3 = allFiles[2*Npos/3 : Npos]

    return(filesPart1, filesPart2, filesPart3)

################################################################################################

# Given a string, this function replaces all the non-alphabetic characters with spaces
# and converts everything to lower case. Also populates the unique words set. Unique words
# are populated only for training docs not for testing.
def words(text, append = 1):

    text = re.sub('[^a-zA-Z\ ]+', " ", text)
    allWords = list(text.split())
    global uniqueWordsSet

    words = []
    for w in allWords:
        if w.lower() not in stopWords:
            if len(w) > 1:
                words.append(w.lower())
                if append > 0:
                    uniqueWordsSet.add(w)

    return words
################################################################################################

# Given a filename, this function returns all its contents in one big string.
def get_text(fileName):

    fileOpen = open(fileName, "r")
    txt = fileOpen.read()

    return txt
################################################################################################


#path = "/Users/Rajendra/USF-Data/EDA-593/review_polarity/txt_sentoken"

# Step 1: Getting the directory name from command line and splitting the
#         pos and neg files into training and testing sets.

# You can give path manually as above and comment the below 2 lines
# or provide the path from command line. If some other person
# is testing the code then the directory should be given in command line.
# Go to Run->Edit Configurations and in Script parameters give the directory name as '-d filename'.

args = parseArgument()
path = args['d'][0]

pathPos = path + "/pos/*.txt"
pathNeg = path + "/neg/*.txt"

allPosFiles = filelist(pathPos)
allNegFiles = filelist(pathNeg)

random.shuffle(allPosFiles)
random.shuffle(allNegFiles)


(p1, p2, p3) = getTrainingAndTestingFiles(allPosFiles)
(n1, n2, n3) = getTrainingAndTestingFiles(allNegFiles)

# Creating empty dictionaries first
trainingDataPos = coll.OrderedDict()
trainingDataNeg = coll.OrderedDict()


testingDataPos  = coll.OrderedDict()
testingDataNeg  = coll.OrderedDict()

trainingDataPos[1] = p1 + p2
trainingDataPos[2] = p2 + p3
trainingDataPos[3] = p3 + p1

testingDataPos[1] = p3
testingDataPos[2] = p1
testingDataPos[3] = p2

trainingDataNeg[1] = n1 + n2
trainingDataNeg[2] = n2 + n3
trainingDataNeg[3] = n3 + n1

testingDataNeg[1] = n3
testingDataNeg[2] = n1
testingDataNeg[3] = n2

# Step 2: Since we have to repeat the whole process 3 times to find accuracies, looping
#         3 times.

for i in range(1,4):

    # Get all words in all 'pos' documents into one list
    allWordsPos = []
    for f in trainingDataPos[i]:
        txt = get_text(f)
        wordsInOneFile = words(txt)
        allWordsPos += wordsInOneFile

    # Get all words in all 'neg' documents into one list
    allWordsNeg = []
    for f in trainingDataNeg[i]:
        txt = get_text(f)
        wordsInOneFile = words(txt)
        allWordsNeg += wordsInOneFile

    # Make dictionaries of 'pos' words and 'neg' words with counters
    dictPosWords = coll.Counter(allWordsPos)
    dictNegWords = coll.Counter(allWordsNeg)

    # This is an important step; Put all the 'pos' words in one set and
    # all the 'neg' words in one set. This is required because when searching
    # for word presence in a particular test document, iterating over dictionary
    # is very very slow compared to iterating over a set.
    posWordsSet = set()
    negWordsSet = set()

    for key1 in dictPosWords.iterkeys():
        posWordsSet.add(key1)

    for key1 in dictNegWords.iterkeys():
        negWordsSet.add(key1)

    PbPosFiles = float(len(allPosFiles)) / (len(allPosFiles) + len(allNegFiles))
    PbNegFiles = float(len(allNegFiles)) / (len(allPosFiles) + len(allNegFiles))

    numCorrectPos = 0
    numInCorrectPos = 0
    numCorrectNeg = 0
    numIncorrectNeg = 0

    Dr = len(allWordsPos) + len(uniqueWordsSet) + 1  # Denominator value for 'pos' docs in the Bayes formula
    Dr1 = len(allWordsNeg) + len(uniqueWordsSet) + 1 # Denominator value for 'neg' docs in the Bayes formula

    # For each test document, calculate the class score using Bayes formula
    # and apply a log transformation and collect correctly and incorrectly
    # predicted documents in their assigned counters.
    for f in testingDataPos[i]:

        txt = get_text(f)
        wordsInOneSampleFile = words(txt, 0)

        classScorePos = math.log(PbPosFiles)
        classScoreNeg = math.log(PbNegFiles)

        for w in wordsInOneSampleFile:

            if w in posWordsSet:
                classScorePos += math.log(float(dictPosWords[w] + 1) / float(Dr))
            else:
                classScorePos += math.log(1 / float(Dr)) #To account for words not found earlier.

            if w in negWordsSet:
                classScoreNeg += math.log(float(dictNegWords[w] + 1) / float(Dr1))
            else:
                classScoreNeg += math.log(1 / float(Dr1))

        if (classScorePos > classScoreNeg):
            numCorrectPos += 1
        else:
            numInCorrectPos += 1


    for f in testingDataNeg[i]:

        txt = get_text(f)
        wordsInOneSampleFile = words(txt, 0)

        classScorePos = math.log(0.5)
        classScoreNeg = math.log(0.5)

        for w in wordsInOneSampleFile:

            if w in posWordsSet:
                classScorePos += math.log(float(dictPosWords[w] + 1) / float(Dr))
            else:
                classScorePos += math.log(1 / float(Dr))

            if w in negWordsSet:
                classScoreNeg += math.log(float(dictNegWords[w] + 1) / float(Dr1))
            else:
                classScoreNeg += math.log(1 / float(Dr1))

        if (classScorePos > classScoreNeg):
            numIncorrectNeg += 1
        else:
            numCorrectNeg += 1


    uniqueWordsSet.clear()
    numAllFiles = numCorrectPos + numCorrectNeg + numInCorrectPos + numIncorrectNeg
    accuracy = 100 * (numCorrectPos + numCorrectNeg) / float(numAllFiles)

    print "*************************************************"
    print " iteration             :", i
    print " num_pos_test_docs     :", len(testingDataPos[i])
    print " num_pos_training_docs :", len(trainingDataPos[i])
    print " num_pos_correct_docs  :", numCorrectPos
    print " num_neg_test_docs     :", len(testingDataNeg[i])
    print " num_neg_training_docs :", len(trainingDataNeg[i])
    print " num_neg_correct_docs  :", numCorrectNeg
    print " accuracy              :", accuracy

################################################################################################
