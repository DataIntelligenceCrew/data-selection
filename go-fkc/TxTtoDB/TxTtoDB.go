package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

/**
Given adjacency list in a text file, load it onto MongoDB.
The text file has format:
index : { neighbor, neighbor, ..., neighbor}
Group assignment is given by another text file, of format:
index : group
The last line of the text files should be an empty line to avoid EOF errors.
*/

type Point struct {
	ID        primitive.ObjectID `bson:"_id,omitempty"`
	Index     int                `bson:"index"`
	Group     int                `bson:"group"`
	Neighbors map[int]bool       `bson:"neighbors"`
}

func main() {
	// Parse flags
	db := flag.String("db", "dummydb", "Name of MongoDB database")
	col := flag.String("col", "dummycol", "Name of MongoDB collection")
	adjFileName := flag.String("adjfile", "test1.txt", "File containing adjacency lists")
	groupFileName := flag.String("groupfile", "test2.txt", "File containing group assignments")
	batchSize := flag.Int("batch", 1000, "DB batch size")
	n := flag.Int("n", 50000, "Number of points in dataset")
	defaultValue := flag.Bool("defaultvalue", true, "The more common value in sparse vector")
	flag.Parse()

	// Access DB & files
	collection, client := getMongoCollection(*db, *col)
	adjFileScanner, adjFile := getFileScanner(*adjFileName)
	groupFileScanner, groupFile := getFileScanner(*groupFileName)
	defer client.Disconnect(context.Background())
	defer adjFile.Close()
	defer groupFile.Close()

	insertIntoCollection(collection, adjFileScanner, groupFileScanner, *batchSize, *n, *defaultValue)
	createIndex(collection)
}

func getMongoCollection(db string, col string) (*mongo.Collection, *mongo.Client) {
	client, err := mongo.Connect(context.Background(), options.Client().ApplyURI("mongodb://localhost:27017"))
	handleError(err)

	database := client.Database(db)
	collection := database.Collection(col)

	_, err = collection.DeleteMany(context.Background(), bson.M{})
	handleError(err)
	return collection, client
}

func getFileScanner(fileName string) (*bufio.Reader, *os.File) {
	file, err := os.Open(fileName)
	handleError(err)

	reader := bufio.NewReader(file)

	//scanner := bufio.NewScanner(file)
	return reader, file
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func parseAdjLine(scanner *bufio.Reader, n int) (int, map[int]bool) {
	ints := make(map[int]bool, 0)
	index := 0
	for {
		line, err := scanner.ReadString('\n')
		if err != nil {
			return -1, map[int]bool{}
		}
		split := strings.Split(line, " : ")
		index, _ = strconv.Atoi(split[0])
		if len(split) > 1 {
			line = split[1]
		}
		line = strings.TrimSpace(line)
		breakThisLoop := false
		if strings.HasSuffix(line, "}") {
			breakThisLoop = true
		}
		line = strings.Trim(line, "{ }\n")
		split = strings.Split(line, ", ")
		for _, v := range split {
			//fmt.Println(v)
			n, _ := strconv.Atoi(v)
			ints[n] = true
		}
		if breakThisLoop {
			break
		}
	}
	return index, ints
}

func parseGroupLine(scanner *bufio.Reader, index int) int {
	for {
		line, err := scanner.ReadString('\n')
		handleError(err)
		line = strings.Trim(line, "\n")
		parts := strings.Split(line, " : ")
		if len(parts) < 2 {
			parts = strings.Split(line, ",")
		}
		i, err := strconv.Atoi(parts[0])
		if err != nil { // Skip any metadata lines
			continue
		}
		if i == index {
			gr, err := strconv.Atoi(parts[1])
			handleError(err)
			return gr
		}
	}
}

func insertIntoCollection(collection *mongo.Collection, adjFileScanner *bufio.Reader, groupFileScanner *bufio.Reader, batchSize int, n int, defaultValue bool) {
	// Read in first line
	nextNonEmptyIndex, nextNonEmptyAdjList := parseAdjLine(adjFileScanner, n)
	fmt.Println(nextNonEmptyIndex, nextNonEmptyAdjList)
	// Outer loop is for batches
	for i := 0; true; i++ { // i = batch number
		points := make([]interface{}, batchSize)
		// Inner loop is within a batch
		for j := 0; j < batchSize; j++ { // j = index within a batch
			k := i*batchSize + j        // k = index of row
			if k != nextNonEmptyIndex { // index k is only covered by itself
				point := Point{
					Index:     k,
					Group:     parseGroupLine(groupFileScanner, k),
					Neighbors: map[int]bool{k: true},
				}
				fmt.Println(point)
				points[j] = point
			} else { // index k has other neighbors
				point := Point{
					Index:     nextNonEmptyIndex,
					Group:     parseGroupLine(groupFileScanner, k),
					Neighbors: nextNonEmptyAdjList,
				}
				fmt.Println(point)
				points[j] = point
				nextNonEmptyIndex, nextNonEmptyAdjList = parseAdjLine(adjFileScanner, n)
				fmt.Println(nextNonEmptyIndex, nextNonEmptyAdjList)
			}
			// Insert batch into mongo
			if j == batchSize-1 || !hasNext(adjFileScanner, groupFileScanner) {
				_, err := collection.InsertMany(context.Background(), points[:j+1])
				handleError(err)
				fmt.Printf("\rBatch %d complete.", i)
			}
		}
		fmt.Printf("\n")
	}
}

func createIndex(collection *mongo.Collection) {
	indexModel := mongo.IndexModel{
		Keys: bson.M{
			"index": 1,
		},
	}
	_, err := collection.Indexes().CreateOne(context.Background(), indexModel)
	handleError(err)
}

func hasNext(adjFileScanner *bufio.Reader, groupFileScanner *bufio.Reader) bool {
	_, err1 := adjFileScanner.Peek(1)
	_, err2 := groupFileScanner.Peek(1)
	if err1 == io.EOF || err2 == io.EOF {
		return false
	} else {
		return true
	}
}
