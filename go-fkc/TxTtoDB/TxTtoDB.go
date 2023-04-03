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
	Neighbors []bool             `bson:"neighbors"`
}

func main() {
	// Parse flags
	db := flag.String("db", "dummydb", "Name of MongoDB database")
	col := flag.String("col", "dummycol", "Name of MongoDB collection")
	adjFileName := flag.String("adjfile", "test1.txt", "File containing adjacency lists")
	groupFileName := flag.String("groupfile", "test2.txt", "File containing group assignments")
	batchSize := flag.Int("batch", 1000, "DB batch size")
	n := flag.Int("n", 50000, "total number of points in dataset")
	flag.Parse()

	// Access DB & files
	collection, client := getMongoCollection(*db, *col)
	adjFileScanner, adjFile := getFileScanner(*adjFileName)
	groupFileScanner, groupFile := getFileScanner(*groupFileName)
	defer client.Disconnect(context.Background())
	defer adjFile.Close()
	defer groupFile.Close()

	insertIntoCollection(collection, adjFileScanner, groupFileScanner, *batchSize, *n)
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

func parseAdjLine(scanner *bufio.Reader, n int) []bool {
	ints := make([]int, 0)
	for {
		line, err := scanner.ReadString('\n')
		handleError(err)
		split := strings.Split(line, " : ")
		if len(split) > 1 { // Index on left side
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
			n, _ := strconv.Atoi(v)
			ints = append(ints, n)
		}
		if breakThisLoop {
			break
		}
	}
	matrix := make([]bool, n)
	for i := 0; i < n; i++ {
		matrix[i] = false
	}
	for i := 0; i < len(ints); i++ {
		matrix[ints[i]] = true
	}
	return matrix
}

func parseGroupLine(scanner *bufio.Reader) int {
	line, err := scanner.ReadString('\n')
	handleError(err)
	line = strings.Trim(line, "\n")
	parts := strings.Split(line, " : ")
	i, err := strconv.Atoi(parts[1])
	handleError(err)
	return i
}

func insertIntoCollection(collection *mongo.Collection,
	adjFileScanner *bufio.Reader, groupFileScanner *bufio.Reader, batchSize int, n int) {
	// Iterate over line of files
	for i := 0; true; i++ {
		if !hasNext(adjFileScanner, groupFileScanner) {
			return
		}

		points := make([]interface{}, batchSize)
		for j := 0; j < batchSize; j++ {
			adjList := parseAdjLine(adjFileScanner, n)
			group := parseGroupLine(groupFileScanner)
			point := Point{
				Index:     batchSize*i + j,
				Group:     group,
				Neighbors: adjList,
			}
			points[j] = point

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
