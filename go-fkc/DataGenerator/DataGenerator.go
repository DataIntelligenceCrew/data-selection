package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

/**
Generate random points in the unit d-dimensional cube to construct random graph.
Then, saves generated graph as CSV and MongoDB collection.
*/

// A point with its group and its coordinates, and its neighbors' indices
type Point struct {
	group int
	coord []float64
}

// A point's group and the indices of its neighbors
// Essentially the same content that will be stored in MongoDB
type PointNeighbors struct {
	ID        primitive.ObjectID `bson:"_id,omitempty"`
	Index     int                `bson:"index"`
	Group     int                `bson:"group"`
	Neighbors []bool             `bson:"neighbors"`
}

func main() {
	// Get command-line flags
	n := flag.Int("n", 1000, "Number of points generated")
	d := flag.Int("d", 3, "Number of dimensions of the hypercube")
	m := flag.Int("m", 5, "Number of distinct groups")
	r := flag.Float64("r", 0.2, "Distance threshold for adjacency")
	db := flag.String("db", "dummydb", "Name of MongoDB database")
	flag.Parse()

	graphID := getGraphID(*n, *d, *m, *r)
	fmt.Println("graphID: ", graphID)
	points := generatePoints(*n, *d, *m)
	adjList := adjacencyList(points, *r)
	//printStats(adjList, *n, *m)
	storeMongo(adjList, *db, graphID)
}

// Returns a unique string identifier for the graph's parameters
func getGraphID(n int, d int, m int, r float64) string {
	str := "n" + strconv.Itoa(n) + "d" + strconv.Itoa(d) + "m" + strconv.Itoa(m)
	return str + "r" + strconv.Itoa(int(r*100))
}

func generatePoints(n int, d int, m int) []Point {
	points := make([]Point, n) // Slice of points
	// Iterate for each point
	for i := 0; i < n; i++ {
		// Create point w/ random group assignment
		point := Point{
			group: rand.Intn(m),
			coord: make([]float64, d),
		}
		// Generate random numbers into each coordinate
		for j := 0; j < d; j++ {
			point.coord[j] = rand.Float64()
		}
		points[i] = point // Add point to slice of points
	}
	return points
}

func adjacencyList(points []Point, r float64) []PointNeighbors {
	n := len(points)
	adjList := make([]PointNeighbors, n)
	for i := 0; i < n; i++ {
		point := points[i]
		adj := PointNeighbors{
			Index:     i,
			Group:     point.group,
			Neighbors: make([]bool, n),
		}
		for j := 0; j < n; j++ {
			other := points[j]
			adj.Neighbors[j] = dist(point.coord, other.coord) <= r
		}
		adjList[i] = adj
	}
	return adjList
}

func printStats(adjList []PointNeighbors, n int, m int) {
	groupCounts := make([]int, m)
	neighborCounts := make([]int, n)
	for i := range adjList {
		adj := adjList[i]
		group := adj.Group
		groupCounts[group]++
		neighborCounts[i] = len(adj.Neighbors)
	}
	fmt.Printf("Group counts: %v\n", groupCounts)
	sort.Ints(neighborCounts)
	fmt.Printf("Min neighbors: %d\n", neighborCounts[0])
	fmt.Printf("Max neighbors: %d\n", neighborCounts[n-1])
	fmt.Printf("Median neighbors: %d\n", neighborCounts[n/2])
}

func storeMongo(adjList []PointNeighbors, db string, graphID string) {
	// Connect to MongoDB
	client, err := mongo.NewClient(options.Client().ApplyURI("mongodb://localhost:27017"))
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = client.Connect(ctx)
	handleError(err)
	defer client.Disconnect(ctx)

	// Open database
	database := client.Database(db)

	// Create collection
	collection := database.Collection(graphID)

	// If the collection is already nonempty, then empty its contents
	_, err = collection.DeleteMany(context.Background(), bson.M{})
	handleError(err)

	// Insert entries into collection
	for i := range adjList {
		_, err = collection.InsertOne(context.Background(), adjList[i])
		handleError(err)
	}

	// Define the index model for the Index field
	indexModel := mongo.IndexModel{
		Keys: bson.M{
			"index": 1,
		},
	}

	// Create the index on the collection
	_, err = collection.Indexes().CreateOne(context.Background(), indexModel)
	handleError(err)
}

// Euclidean distance between two points
func dist(foo []float64, bar []float64) float64 {
	d := len(foo)
	sumSquares := 0.0
	for i := 0; i < d; i++ {
		diff := foo[i] - bar[i]
		sumSquares += diff * diff
	}
	return math.Sqrt(sumSquares)
}

// Very basic error logging
func handleError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
