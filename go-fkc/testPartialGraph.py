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
        #slices = np.arange(0, len(X_train), dtype=int)
        #X_train, X_test, Y_train, Y_test= train_test_split(X, Y, trainSize=0.8)

        #load the full data adj matrix into mongo
        selector1 = FairFacilityLocation(colName, 0.1, 10, "testing123", 10, dset=X, groupLabels=Y, reuseMongo=False, iterPrint=True) # mongoDBname=dbName, slices=slice_train)



# Set a random seed (you can use any integer value)
seed_value = 42
np.random.seed(seed_value)

fullData = np.random.random((100, 10))
groupLabels = np.random.randint(0, 10, size=(100))
x = PartialGraphTester("testing1234", fullData, groupLabels)




    

