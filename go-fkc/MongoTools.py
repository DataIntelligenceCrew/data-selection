#import pymongo
from pymongo import MongoClient


class MongoCollection:
    def __init__(self, dbName, collectionName, connectionString="mongodb://localhost:27017"):
        self.connectionString = connectionString
        self.dbName = dbName
        self.collectionName = collectionName

        self.collection = self.getCollection() 

    def getDatabase(self):
        client = MongoClient(self.connectionString)
    

        return client[self.dbName]

    def getCollection(self):
        return self.getDatabase()[self.collectionName]

    def insertIntoCollection(self, index, group, neighbors):

        item = {
            "index" : index,
            "group" : group,
            "neighbors" : neighbors
        }

        self.collection.insert_one(item)
    def hasElements(self):
        return len(list(self.collection.find())) != 0
    def empty(self):
        self.collection.delete_many({})





