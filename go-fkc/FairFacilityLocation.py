from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from MongoTools import MongoCollection
from MongoGraphLoader import MongoGraphLoader
import random
import os

class FairFacilityLocation:
    def __init__(self, mongoColName, groupReq, groupCount, ExperimentID, coresetSize, slices=None, dset=None, groupLabels=None, mongoDBname="MichaelFlynn", reuseMongo = True, numThreads = 1, optimization = "Lazy", iterPrint = False):

        #IMPORTANT: if you load a new mongo database, it will return the indicies in X_train to use
        #If you use a partial graph with slices, it will return the indicies in X to use
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

        if slices is None:
            self.partialGraph = False
        else:
            self.partialGraph = True
        #self.partialGraph = True if slices != None else False

        if not self.partialGraph:
            self.slices = []
        else:
            self.slices = slices
        
        self.ssSize = len(self.slices)
        self.coresetIndicies = []
        self.CSVLocation = os.path.join(os.getcwd(), "configs", "facilityConfigs", str(self.ExperimentID) + ".csv")

        self.SAVE_TO_LOCATION = str(os.path.join(os.getcwd(), "coresets", "facility")) #where coreset will be written

        #prepare mongo database
        if not reuseMongo:
            print("creating new collection")
            collection = MongoGraphLoader(dset, groupLabels, mongoDBname, mongoColName).getCollection()
        else:
            print("re-using collection with or without slices")
            collection = MongoCollection(mongoDBname, mongoColName)
            if not collection.hasElements():
                print("ERROR: wanted to reuse collection but collection specified is empty")


        print("writing csv file for submdoular cover")
        self.writeCSV()
        print("preforming data selection")
        self.preformDataSelection()

    
    def getCoreset(self):

        coresetLocation = os.path.join(self.SAVE_TO_LOCATION, str(self.ExperimentID) + ".txt")
        with open(coresetLocation, 'r+') as file:
            lines = file.read().split("\n")
            print(lines)
            numbers = lines[7:len(lines)-1]
            print(numbers)

            for number in range(len(numbers)):
                if not self.partialGraph:
                    numbers[number] = int(numbers[number])
                else:
                    numbers[number] = sorted(self.slices)[int(numbers[number])]


        self.coresetIndicies = numbers
        return numbers



    def writeCSV(self):
        prefix = "DB,Collection,GroupReq,GroupCnt,Optim,Threads,Cardinality,IterPrint,ResultDest,ID,partialGraph,slices,ssSize\n"

        fileText = str(prefix) + self.dbName + "," + self.colName + \
                "," + str(self.groupReq) + "," + str(self.groupCount) + \
                "," + str(self.optimization) + "," + str(self.numThreads) + \
                "," + str(self.coresetSize) + "," + str(self.iterPrint).lower() + \
                "," + str(self.SAVE_TO_LOCATION) + "," + str(self.ExperimentID) + \
                "," + str(self.partialGraph).lower() + "," + self.prettySlices(self.slices) + "," + str(self.ssSize) + "\n"
        with open(self.CSVLocation, 'w+') as file:
            file.write(fileText)

    def prettySlices(self, slices):
        string = ""

        for slice in slices:
            string += str(slice) + " "
        
        string = string[0:len(string)-1]
        return string
        
    def preformDataSelection(self):

        #for server experiments
        #changeToDirectory = "cd " + os.path.join(os.getcwd(), "FairFacilityLocation", "FairFacilityLoc") 

        changeToDirectory = os.path.join(os.getcwd(), "data-selection", "go-fkc")
        process = "go build .\FairFacilityLoc"
        run = "FairFacilityLoc.exe -config=" + self.CSVLocation
    
        print("CD: " + changeToDirectory)
        print("process: " + process)
        print("run: " + run)

        os.chdir(changeToDirectory)
        os.system(process)
        os.system(run)
        print("done")



    
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

          
