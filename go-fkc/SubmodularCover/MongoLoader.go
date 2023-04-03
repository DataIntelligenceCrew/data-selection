package main

import (
	"context"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

/**
Importing a MongoDB Collection.
*/

func getMongoCollection(dbName string, collectionName string) *mongo.Collection {
	// Create mongoDB server connection
	clientOptions := options.Client().ApplyURI("mongodb://localhost:27017")
	client, err := mongo.Connect(context.Background(), clientOptions)
	handleError(err)

	// Create handles
	db := client.Database(dbName)
	collection := db.Collection(collectionName)

	return collection
}

func getFullCursor(collection *mongo.Collection) *mongo.Cursor {
	cur, err := collection.Find(context.Background(), bson.M{})
	handleError(err)
	return cur
}

func getEntryFromCursor(cur *mongo.Cursor) Point {
	var entry Point
	err := cur.Decode(&entry)
	handleError(err)
	return entry
}

func getSetCursor(collection *mongo.Collection, set map[int]bool) *mongo.Cursor {
	return getSliceCursor(collection, mapToSlice(set))
}

func getSliceCursor(collection *mongo.Collection, slice []int) *mongo.Cursor {
	filter := bson.M{
		"index": bson.M{
			"$in": slice,
		},
	}
	return getCursorFilter(collection, filter)
}

func getRangeCursor(collection *mongo.Collection, min int, max int) *mongo.Cursor {
	filter := bson.M{
		"index": bson.M{
			"$gte": min,
			"$lte": max,
		},
	}
	return getCursorFilter(collection, filter)
}

func getCursorFilter(collection *mongo.Collection, filter bson.M) *mongo.Cursor {
	cur, err := collection.Find(context.Background(), filter)
	handleError(err)
	return cur
}

func getCollectionSize(collection *mongo.Collection) int {
	count, err := collection.CountDocuments(context.Background(), bson.D{})
	handleError(err)
	return int(count)
}

func getPointFromDB(collection *mongo.Collection, index int) Point {
	filter := bson.M{"index": index}
	cur, err := collection.Find(context.Background(), filter)
	handleError(err)
	defer cur.Close(context.Background())
	var p Point
	cur.Next(context.Background())
	err = cur.Decode(&p)
	handleError(err)
	return p
}

/**
Test method for this file's methods.
*/
/*
func main() {
	dbName := "dummydb"
	collectionName := "n1000d3m5r20"
	collection := getMongoCollection(dbName, collectionName)
	graph := getFullGraph(collection)
	fmt.Printf("%v\n", graph)
}
*/
