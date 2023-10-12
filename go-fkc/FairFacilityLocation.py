from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from MongoTools import MongoCollection
import random
import os

class FairFacilityLocation:
    def __init__(self, mongoColName, groupReq, groupCount, ExperimentID, coresetSize, dset=None, labels=None, groupLabels=None, mongoDBname="MichaelFlynn", reuseMongo = True, numThreads = 1, optimization = "Lazy", iterPrint = False):
        print("Fair facility DSET SIZE IS: " + str(len(dset)))
        self.dbName = mongoDBname
        self.colName = mongoColName
        self.groupReq = groupReq
        self.reuseMongo = reuseMongo
        self.numThreads = numThreads
        self.groupCount = groupCount
        self.optimization = optimization
        self.coresetSize = coresetSize
        self.iterPrint = iterPrint
        self.ExperimentID = ExperimentID
        self.coresetIndicies = []
        self.CSVLocation = os.path.join(os.getcwd(), "configs", "facilityConfigs", str(self.ExperimentID) + ".csv")

        self.SAVE_TO_LOCATION = str(os.path.join(os.getcwd(), "coresets", "facility")) #where coreset will be written

        #prepare mongo database
        if not reuseMongo:

            print("creating mongoDB")
            print("Checking if specified collection is empty")
            collection = MongoCollection(mongoDBname, mongoColName)
            if collection.hasElements():
                collection.empty()

                print("Collection not found")
                print("Creating adjacecny matrix")
                #construct adjacencyMatrix
                adjMatrix = dict()

                for i in range(len(dset)):
                    adjMatrix.update({i: []})
                    for j in range(len(dset)):

                        #print(type(dset))
                        adjMatrix[i].append(self.sim(dset[i, :], dset[j, :]))
                
                self.adjMatrix = adjMatrix

                print("adjacency matrix constructed")
                print("ADJ MATRIX SIZE: " + str(len(adjMatrix)))
                print("num keys in adj matrix:  " + str(len(adjMatrix.keys())))

                for i in adjMatrix.keys():
                    collection.insertIntoCollection(int(i), int(groupLabels[i]), adjMatrix[i])

        else:

        
        
        #make a call to FairFacilityLocation, passing it hyperparmeters and mongo collection

        print("writing csv file for submdoular cover")
        self.writeCSV()
        print("preforming data selection")
        self.preformDataSelection()

    
    def getCoreset(self):

        coresetLocation = os.path.join(self.SAVE_TO_LOCATION, str(self.ExperimentID) + ".txt")
        with open(coresetLocation, 'r+') as file:
            lines = file.read().split("\n")
            coresetLine = lines[len(lines)-1]
            numbers = coresetLine.split(": ")[1].split(",")
            numbers = numbers[0:len(numbers)-1]

            for number in range(len(numbers)):
                numbers[number] = int(numbers[number])

        self.coresetIndicies = numbers
        return numbers



    def writeCSV(self):
        prefix = "DB,Collection,GroupReq,GroupCnt,Optim,Threads,Cardinality,IterPrint,ResultDest,ID\n"

        fileText = str(prefix) + self.dbName + "," + self.colName + "," + str(self.groupReq) + "," + str(self.groupCount) + "," + str(self.optimization) + "," + str(self.numThreads) + "," + str(self.coresetSize) + "," + str(self.iterPrint).lower() + "," + str(self.SAVE_TO_LOCATION) + "," + str(self.ExperimentID) + "\n"
        with open(self.CSVLocation, 'w+') as file:
            file.write(fileText)
        
    def preformDataSelection(self):

        changeToDirectory = "cd " + os.path.join(os.getcwd(), "FairFacilityLocation", "FairFacilityLoc") 
        process = "go run *.go -config=" + self.CSVLocation
    
        os.system(changeToDirectory + ";" + process)


    #cosine similarity between feature vectors a and b
    def sim(self, a, b):
        cos_similarity = np.dot(a, b)/(norm(a)*norm(b))
        if cos_similarity == 1:
            return 1.0
        return float(cos_similarity)
    
    def utility(self, D, S):

        utility = 0
        for d in D:
            closestSim = -1
            for s in S:
                if self.similarity(d, s) > closestSim:
                    closestSim = self.similarity(d,s)
            utility += closestSim


    


def test(m, n):
    dset = np.random.random((m, n))
    labels = [random.randint(0, 9) for i in range(m)]

    foo = FairFacilityLocation(dset, labels, labels, "testing", 10, "testing123", 2 )
    print(foo.getCoreset())

def unitTest(foo):
    for i in foo.adjMatrix.keys():
        counter = 0
        
        for j in foo.adjMatrix[i]:
            if j == foo.sim(foo.dset[i, :], foo.dset[counter, :]):
                print("good")
            counter += 1



def functionValueTest():

    x = FairFacilityLocation(None, None, None,"adjfile" ,-1,"functionValueTest")
    coreset = x.getCoreset()



    #Want to run the following experiment:
    #Take a fairFacil coreset
        #get its objective value
    #take a random subset
        #get its objective value

    #just get a fairFacil coreset

    #duck tape jiwons code to evaluate the random subset for utility







#test(5, 2)

          
