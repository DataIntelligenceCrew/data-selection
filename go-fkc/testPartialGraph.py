from sklearn.model_selection import train_test_split
from FairFacilityLocation import FairFacilityLocation
from Saturate import Saturate
import numpy as np

class PartialGraphTester:
    def __init__(self, dbName, colName, X, Y):

        self.testFairFacility(dbName, colName, X, Y)
        self.testSaturate(dbName, colName, X, Y)

    def testFairFacility(self, X, Y):

        #take a subset of X and Y
        slices = np.arange(0, len(X_train), dtype=int)
        X_train, X_test, Y_train, Y_test, slice_train, slice_test = train_test_split(X, Y, slices, trainSize=0.8)
        #load the subset into a mongoDB
        #take fairFacility coreset of subset db

        #run fairFacility again but using full mongoDB with slices

        #compare the returned coresets


    def testSaturate(self):
        #take a subset of X and Y
        #load the subset into a mongoDB
        #take saturate coreset of subset db

        #run saturate again but using full mongoDB with slices

        #compare the returned coresets
    

