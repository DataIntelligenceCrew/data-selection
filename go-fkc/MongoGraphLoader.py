#loads in full adjacency matrix graphs of datasets into mongo

#FORMAT is:
# index : int
# group : int
# affinities: int[] where affinities[i] is the similarity between point at this index and point at ith index


from MongoTools import MongoCollection

class MongoGraphLoader:

    def __init__(self, dset, groupLabels, dbName, collectionName):
            
        print("loading in mongo database")
        self.collection = MongoCollection(dbName, collectionName)

        if self.collection.hasElements():
            print("The collection you specified is non-empty, empty it?")
            x = input("yes/no")
            if x == "yes":
                print("emptying collection")
                self.collection.empty()
            else:
                print("Error: non-empty collection")
                self.ok = False

        
        print("constructing adjacency matrix")
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
            self.collection.insertIntoCollection(int(i), int(groupLabels[i]), adjMatrix[i])

    def getCollection(self):
        return self.collection
    


