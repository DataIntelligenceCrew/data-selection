from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from MongoTools import MongoCollection
import random
import os

class Saturate:
    def __init__(self, dset, groupLabels, numGroups, ExperimentID, coresetSize, mongoColName, mongoDBname, alpha=1.0, optim="Lazy", numThreads=4, iterPrint=False, reuseMongo=False):
        
        self.dbName = mongoDBname
        self.colName = mongoColName
        self.groupCount = numGroups
        self.optimization = optim
        self.numThreads = numThreads
        self.coresetSize = coresetSize
        self.iterPrint = iterPrint

        self.alpha = alpha
        self.ExperimentID = ExperimentID
        self.CSVLocation = os.path.join(os.getcwd(), "configs", "saturate", str(self.ExperimentID) + ".csv")

        self.SAVE_TO_LOCATION = str(os.path.join(os.getcwd(), "coresets", "saturate")) #where coreset will be written

        #prepare mongo database
        if not reuseMongo:

            print("Finding/creating mongoDB")

            print("Checking if adjMatrix already found")
            collection = MongoCollection(mongoDBname, mongoColName)
            if not collection.hasElements():
            
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
                print("Resuing already created mongo db")
            print("mongoDB loaded")

        self.writeCSV()
        self.preformDataSelection()

    
    def getCoreset(self):

        coresetLocation = os.path.join(self.SAVE_TO_LOCATION, str(self.ExperimentID) + ".txt")
        indicies = []

        with open(coresetLocation, 'r+') as file:
            lines = file.read().split("\n")
            print(lines)

            for i in range(8, len(lines)-1):
                indicies.append(int(lines[i]))

        return indicies

    

    def preformDataSelection(self):

        changeToDirectory = "cd " + os.path.join(os.getcwd(), "SATURATE") 
        process = "go run *.go -config=" + self.CSVLocation
    
        os.system(changeToDirectory + ";" + process)

    def writeCSV(self):
        prefix = "DB,Collection,GroupCnt,Optim,Threads,Cardinality,IterPrint,ResultDest,Alpha,ID\n"

        fileText = str(prefix) + self.dbName + "," + self.colName + "," + str(self.groupCount) + "," + str(self.optimization) + "," + str(self.numThreads) + "," + str(self.coresetSize) + "," + str(self.iterPrint).lower() + "," + str(self.SAVE_TO_LOCATION) + "," + str(self.alpha) + "," + str(self.ExperimentID) + "\n"
        with open(self.CSVLocation, 'w+') as file:
            file.write(fileText)

    #cosine similarity between feature vectors a and b
    def sim(self, a, b):
        cos_similarity = np.dot(a, b)/(norm(a)*norm(b))
        if cos_similarity == 1:
            return 1.0
        return float(cos_similarity)


    def adjMatrix(self, dset):

        adjMatrix = [[0 for i in range(len(dset))] for i in range(len(dset))]
        print(len(adjMatrix))

