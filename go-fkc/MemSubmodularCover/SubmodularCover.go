package main

import (
	"bufio"
	"context"
	"os"
	"strconv"
	"strings"

	//"fmt"

	"database/sql"

	_ "github.com/lib/pq"

	"github.com/bits-and-blooms/bitset"
)

type Graph struct {
	adjMatrix    []*bitset.BitSet
	groups       []int
	numNeighbors []int
}

/*
Optimization modes:
0: Classic greedy
1: Lazy greedy
2: Lazy Lazy greedy
3: Multilevel with lazylazy -> lazy
2: Distributed submodular cover (DisCover) using GreeDi & lazygreedy as subroutines
*/
func SubmodularCover(dbType string, dbName string, collectionName string,
	coverageReq int, groupReqs []int, optimMode int, threads int, cardinality int,
	dense bool, eps float64, print bool, groupFile string) ([]int, int) {
	// Import & Initialize all stuff
	graph, n := getGraph(dbType, dbName, collectionName, groupFile, print)
	coverageTracker, groupReqs, coreset := getTrackers(graph, coverageReq, groupReqs, dense, n)
	initialRemainingScore := remainingScore(coverageTracker, groupReqs)
	decrementAllTrackers(graph, coreset, coverageTracker, groupReqs)
	report("initialized trackers\n", print)

	// Choose algorithm to run
	candidates := setMinus(rangeSet(n), sliceToSet(coreset))
	var result []int
	switch optimMode {
	case 0:
		result = classicGreedy(graph, coverageTracker, groupReqs, coreset, candidates, cardinality, threads, print)
	case 1:
		result = lazyGreedy(graph, coverageTracker, groupReqs, coreset, candidates, cardinality, threads, print, false)
	case 2:
		result = lazyLazyGreedy(graph, coverageTracker, groupReqs, coreset, candidates, cardinality, threads, print, false, eps)
	case 3:
		result = greeDi(graph, coverageTracker, groupReqs, coreset, candidates, cardinality, threads, print, 0, eps)
	case 4:
		result = greeDi(graph, coverageTracker, groupReqs, coreset, candidates, cardinality, threads, print, 1, eps)
	default:
		result = []int{}
	}
	// Compute score of the collected coreset
	finalRemainingScore := remainingScore(coverageTracker, groupReqs)
	functionValue := initialRemainingScore - finalRemainingScore
	return result, functionValue
}

func getGraph(dbType string, dbName string, collectionName string, groupFile string, print bool) (Graph, int) {
	switch dbType {
	case "mongo":
		return getMongoGraph(dbName, collectionName, print)
	case "psql":
		return getPostgresGraph(dbName, collectionName, print)
	default:
		return Graph{}, 0
	}
}

func getMongoGraph(dbName string, collectionName string, print bool) (Graph, int) {
	// Get collection
	collection := getMongoCollection(dbName, collectionName)
	n := getCollectionSize(collection)
	// Initialize results
	graph := Graph{
		adjMatrix:    make([]*bitset.BitSet, n),
		groups:       make([]int, n),
		numNeighbors: make([]int, n),
	}
	// Get cursor for all entries in collection
	cur := getFullCursor(collection)
	defer cur.Close(context.Background())
	for i := 0; cur.Next(context.Background()); i++ {
		point := getEntryFromCursor(cur)
		graph.adjMatrix[i] = setToBitSet(point.Neighbors, n)
		graph.groups[i] = point.Group
		graph.numNeighbors[i] = int(graph.adjMatrix[i].Count())
		report("loading db to memory "+strconv.Itoa(i)+"\r", print)
	}
	report("\n", print)
	return graph, n
}

func getPostgresGraph(dbName string, tableName string, print bool) (Graph, int) {
	// Get cursor
	rows, n := getFullPostgresCursor(dbName, tableName)
	defer rows.Close()
	// Initialize results
	graph := Graph{
		adjMatrix:    make([]*bitset.BitSet, n),
		groups:       make([]int, n),
		numNeighbors: make([]int, n),
	}
	// Parse group txt file
	groupFileScanner, groupFile := getFileScanner(*groupFileName)
	defer groupFile.Close()
	// Iterate over each entry
	for i := 0; rows.Next(); i++ {
		// Adjacent matrix
		var (
			id int
			pl []uint8
		)
		err := rows.Scan(&id, &pl)
		handleError(err)
		graph.adjMatrix[i] = listToBitSet(pl, n)
		// Numneighbors
		graph.numNeighbors[i] = int(graph.adjMatrix[i].Count())
		handleError(rows.Err())
		// group
		gr := parseGroupLine(groupFileScanner, i)
		graph.groups[i] = gr
		report("loading db to memory "+strconv.Itoa(i)+"\r", print)
	}
	report("\n", print)
	return graph, n
}

func getFileScanner(fileName string) (*bufio.Reader, *os.File) {
	file, err := os.Open(fileName)
	handleError(err)
	reader := bufio.NewReader(file)
	//scanner := bufio.NewScanner(file)
	return reader, file
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

func getFullPostgresCursor(dbName string, tableName string) (*sql.Rows, int) {
	//connStr := "postgres://jchang38@localhost/imagenet_1k?sslmode=disable"
	connStr := "host=/tmp user=jchang38 dbname=" + dbName + " sslmode=disable"
	//fmt.Println(connStr)
	db, err := sql.Open("postgres", connStr)
	handleError(err)
	// Grabbing all rows
	rows, err := db.Query("SELECT * FROM " + tableName)
	// Counting elements
	var n int
	err = db.QueryRow("SELECT COUNT(*) FROM " + tableName).Scan(&n)
	handleError(err)
	return rows, n
}

func getTrackers(graph Graph, coverageReq int, groupReqs []int, dense bool, n int) ([]int, []int, []int) {
	if dense {
		coverageTracker := make([]int, n)
		for i := 0; i < n; i++ {
			coverageTracker[i] = coverageReq
		}
		return coverageTracker, groupReqs, []int{}
	} else {
		coverageTracker := make([]int, n)
		coreset := make([]int, 0)
		for i := 0; i < n; i++ {
			coverageTracker[i] = min(graph.numNeighbors[i], coverageReq)
			//if graph.numNeighbors[i] <= coverageReq {
			//	coreset = append(coreset, i)
			//	groupReqs[graph.groups[i]] = max(0, groupReqs[graph.groups[i]]-1)
			//}
		}
		return coverageTracker, groupReqs, coreset
	}
}

func marginalGain(graph Graph, index int, coverageTracker []int, groupTracker []int) int {
	gain := 0
	// Marginal gain from coverage req
	neighborVec := graph.adjMatrix[index]
	for neighbor, ok := neighborVec.NextSet(0); ok; neighbor, ok = neighborVec.NextSet(neighbor + 1) {
		gain += coverageTracker[neighbor]
	}
	// Marginal gain from group req
	gain += groupTracker[graph.groups[index]]
	return gain
}

func getMarginalGains(graph Graph, coverageTracker []int,
	groupTracker []int, candidates map[int]bool) []*Item {
	results := make([]*Item, len(candidates))
	i := 0
	for index := range candidates {
		gain := marginalGain(graph, index, coverageTracker, groupTracker)
		item := &Item{
			value:    index,
			priority: gain,
		}
		results[i] = item
		i++
	}
	return results
}

func notSatisfied(coverageTracker []int, groupTracker []int) bool {
	return sum(coverageTracker)+sum(groupTracker) > 0
}

func decrementTrackers(graph Graph, index int, coverageTracker []int, groupTracker []int) {
	// Decrement trackers of neighbors
	neighborVec := graph.adjMatrix[index]
	for neighbor, ok := neighborVec.NextSet(0); ok; neighbor, ok = neighborVec.NextSet(neighbor + 1) {
		coverageTracker[neighbor] = max(0, coverageTracker[neighbor]-1)
	}
	// Decrement tracker for group
	group := graph.groups[index]
	groupTracker[group] = max(0, groupTracker[group]-1)
}

func decrementAllTrackers(graph Graph, points []int, coverageTracker []int, groupTracker []int) {
	for i := 0; i < len(points); i++ {
		decrementTrackers(graph, points[i], coverageTracker, groupTracker)
	}
}

func remainingScore(coverageTracker []int, groupTracker []int) int {
	return sum(coverageTracker) + sum(groupTracker)
}
