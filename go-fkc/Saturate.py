from MongoGraphLoader import MongoGraphLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from MongoTools import MongoCollection
import random
import os

class Saturate:
    def __init__(self, mongoColName, groupReq, numGroups, ExperimentID, coresetSize, slices=None, dset=None, groupLabels=None, mongoDBname="MichaelFlynn", reuseMongo=True, alpha=1.0, optim="Lazy", numThreads=4, iterPrint=True):

        #self, mongoColName, groupReq, groupCount, ExperimentID, coresetSize, slices=None, dset=None, groupLabels=None, mongoDBname="MichaelFlynn", reuseMongo = True, numThreads = 1, optimization = "Lazy", iterPrint = False)
        
        self.dbName = mongoDBname
        self.colName = mongoColName
        self.groupCount = numGroups
        self.optimization = optim
        self.numThreads = numThreads
        self.coresetSize = coresetSize
        self.iterPrint = iterPrint

        if slices is None:
            self.partialGraph = False
        else:
            self.partialGraph = True
        if not self.partialGraph:
            self.slices = []
        else:
            self.slices = slices
    
        self.ssSize = len(self.slices)
        self.alpha = alpha
        self.ExperimentID = ExperimentID
        self.CSVLocation = os.path.join(os.getcwd(), "configs", "saturate", str(self.ExperimentID) + ".csv")

        self.SAVE_TO_LOCATION = str(os.path.join(os.getcwd(), "coresets", "saturate")) #where coreset will be written

        #prepare mongo database
        if not reuseMongo:

            print("creating new collection")
            collection = MongoGraphLoader(dset, groupLabels, mongoDBname, mongoColName).getCollection()

        else:
            print("re-using collection with or without slices")
            collection = MongoCollection(mongoDBname, mongoColName)
            if not collection.hasElements():
                print("ERROR: wanted to reuse collection but collection specified is empty")

        self.writeCSV()
        self.preformDataSelection()

    def prettySlices(self, slices):
        string = ""

        for slice in slices:
            string += str(slice) + " "
        
        string = string[0:len(string)-1]
        return string

    
    def getCoreset(self):

        coresetLocation = os.path.join(self.SAVE_TO_LOCATION, str(self.ExperimentID) + ".txt")
        indicies = []
        sortedSlices = sorted(self.slices)

        with open(coresetLocation, 'r+') as file:
            lines = file.read().split("\n")
            print(lines)

            for i in range(8, len(lines)-1):
                if self.partialGraph:
                    indicies.append(sortedSlices[int(lines[i])])
                else:
                    indicies.append(int(lines[i]))
                    

        return indicies

    def preformDataSelection(self):

        #changeToDirectory = "cd " + os.path.join(os.getcwd(), "SATURATE") 
        #process = "go run *.go -config=" + self.CSVLocation
    
        #os.system(changeToDirectory + ";" + process)
        changeToDirectory = os.path.join(os.getcwd(), "data-selection", "go-fkc")
        process = "go build .\Saturate"
        run = "Saturate.exe -config=" + self.CSVLocation
    
        print("CD: " + changeToDirectory)
        print("process: " + process)
        print("run: " + run)

        os.chdir(changeToDirectory)
        os.system(process)
        os.system(run)
        print("done")

    def writeCSV(self):
        prefix = "DB,Collection,GroupCnt,Optim,Threads,Cardinality,IterPrint,ResultDest,Alpha,ID,partialGraph,slices,ssSize\n"

        fileText = str(prefix) + self.dbName + \
            "," + self.colName + "," + str(self.groupCount) +\
            "," + str(self.optimization) + "," + str(self.numThreads) +\
            "," + str(self.coresetSize) + "," + str(self.iterPrint).lower() +\
            "," + str(self.SAVE_TO_LOCATION) + "," + str(self.alpha) +\
            "," + str(self.ExperimentID) +  "," + str(self.partialGraph).lower() +\
            "," + self.prettySlices(sorted(self.slices)) + "," + str(self.ssSize) + "\n"
        

        with open(self.CSVLocation, 'w+') as file:
            file.write(fileText)





