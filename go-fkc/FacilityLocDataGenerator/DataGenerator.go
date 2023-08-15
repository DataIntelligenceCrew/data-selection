package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
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
	ID         primitive.ObjectID `bson:"_id,omitempty"`
	Index      int                `bson:"index"`
	Group      int                `bson:"group"`
	Affinities []float64          `bson:"neighbors"`
}

func main() {
	// Get command-line flags
	n := flag.Int("n", 1000, "Number of points generated")
	d := flag.Int("d", 3, "Number of dimensions of the hypercube")
	m := flag.Int("m", 5, "Number of distinct groups")
	db := flag.String("db", "dummydb", "Name of MongoDB database")
	flag.Parse()

	graphID := getGraphID(*n, *d, *m)
	fmt.Println("graphID: ", graphID)
	points := generatePoints(*n, *d, *m)
	fmt.Println("generated points")
	storeMongo(points, *db, graphID)
}

// Returns a unique string identifier for the graph's parameters
func getGraphID(n int, d int, m int) string {
	return "n" + strconv.Itoa(n) + "d" + strconv.Itoa(d) + "m" + strconv.Itoa(m)
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

func storeMongo(points []Point, db string, graphID string) {
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
	n := len(points)
	dim := len(points[0].coord)
	maxDist := math.Sqrt(float64(dim) * 1.0)
	for i := 0; i < n; i++ {
		point := points[i]
		pointNeighbor := &PointNeighbors{
			Index: i,
			Group: point.group,
			Affinities: make([]float64, n),
		}
		for j := 0; j < n; j++ {
			sim := maxDist - dist(points[i].coord, points[j].coord)
			pointNeighbor.Affinities[j] = sim
		}
		_, err = collection.InsertOne(context.Background(), pointNeighbor)
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
