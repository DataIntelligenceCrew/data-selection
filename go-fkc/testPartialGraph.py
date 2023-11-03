from MongoGraphLoader import MongoGraphLoader
from sklearn.model_selection import train_test_split
from FairFacilityLocation import FairFacilityLocation
from Saturate import Saturate
import numpy as np

class PartialGraphTester:
    def __init__(self, colName, X, Y):

        self.testFairFacility( colName, X, Y)
        #self.testSaturate(dbName, colName, X, Y)

    def testFairFacility(self, colName, X, Y):

        #1) make sure the csv file is getting written correctly
        slices = np.arange(0, len(X), dtype=int)
        X_train, X_test, Y_train, Y_test, slice_train, slice_test = train_test_split(X, Y, slices, train_size=0.8)
        #print(X_train.shape)
        #self.prettySlices(slice_test)
        #1) load the full data adj matrix into mongo DONE

        #2) get a coreset using only slices of the full adj matrix
        selector1 = FairFacilityLocation(colName, -1, 10, "halfMAtrix", 10, dset=X_train, groupLabels=Y_train, reuseMongo=False, iterPrint=True) # mongoDBname=dbName, slices=slice_train)
        #print(selector1.getCoreset())
        # 







        #selector1 = Saturate("saturateTest", -1, 10, "fullX_train", 10, dset=X_train, groupLabels=Y_train, reuseMongo=False)
        #print("Coreset from X_train graph: " + str(selector1.getCoreset())) #[37, 27, 30, 43, 0, 8, 68, 19, 61, 36]

        #MongoGraphLoader(X, Y, "MichaelFlynn", "partialGraphTestSaturate")
        #selector2 = Saturate("partialGraphTestSaturate", -1, 10, "partialGraph", 10, slices=slice_train, reuseMongo=True) 
        #print("Coreset from partial graph: " + str(selector2.getCoreset()))
        #partialGResults = [27, 48, 68, 36, 71, 22, 19, 63, 47, 2]
        #newMongoResults = [37, 55, 10, 43, 68, 0, 67, 9, 8, 61]


        partialGResults = [27, 39, 41, 36, 22, 47, 71, 7, 2, 28]
        newMongoResults = [37, 27, 30, 43, 0, 8, 68, 19, 61, 36]

        #? are these actually the same if I take the first from full dataset but second from X_train
        
        
        partialGResults = X[partialGResults]
        newMongoResults = X_train[newMongoResults]

        for item in partialGResults:
            if item not in newMongoResults:
                print("error")
            else:
                print("success")


        #selector2 = FairFacilityLocation("testing45678910", -1, 10, "testing456789", 10, dset=X_train, groupLabels=Y_train, reuseMongo=False, iterPrint=True) # mongoDBname=dbName, slices=slice_train)
        #print(selector2.getCoreset())








# Set a random seed (you can use any integer value)
seed_value = 42
np.random.seed(seed_value)

fullData = np.random.random((100, 10))
groupLabels = np.random.randint(0, 10, size=(100))
x = PartialGraphTester("halfMAtrix", fullData, groupLabels)




    

