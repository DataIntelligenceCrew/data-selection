package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
)

type Graph struct {
	affinityMatrix [][]float64
	groups         []int
	n              int
}

func (this Graph) getSim(i int, j int) float64 {

	print("i, j sim = ")

	// if i=0 and j=0 return 1.0
	// if i=0 and j=1 return matrix[i][j-1]

	//if i=1 and j=0 return matrix[]
	if i < j {
		print(fmt.Sprintf("%f", this.affinityMatrix[i][j-1-i]) + "\n")
		return this.affinityMatrix[i][j-1-i]
	} else if i > j {
		print(fmt.Sprintf("%f", this.affinityMatrix[j][i-1-j]) + "\n")
		return this.affinityMatrix[j][i-1-j]
	} else {
		return 1.0
	}
}

/*
Client syntax:
coreset, funcVal, preTime, inTime := SubmodularCover(db, collection, groupReq,

	groupCnt, optim, threads, cardinality, iterPrint)
*/
func SubmodularCover(dbName string, collectionName string, groupReq int,
	groupCnt int, optimMode string, threads int, cardinality int,
	print bool, partialGraph bool, slices []int, ssSize int) ([]int, float64, time.Duration, time.Duration) {
	preTime := time.Now()
	// Import & Initialize all stuff

	var graph Graph

	if partialGraph {
		report("getting partial graph\n", print)
		graph = getPartialGraph(dbName, collectionName, print, ssSize, slices)
	} else {
		report("getting full graph\n", print)

		graph = getGraph(dbName, collectionName, print)
	}

	groupReqTracker := make([]int, groupCnt) // Remaining group cnt reqs
	for i := range groupReqTracker {         // Initialize with groupReq
		if groupReq >= 0 {
			groupReqTracker[i] = groupReq
		} else {
			groupReqTracker[i] = graph.n
		}
	}
	closestInCoreset := make([]int, graph.n) // Index of nearest point in coreset
	for i := 0; i < graph.n; i++ {
		closestInCoreset[i] = -1 // -1 means coreset is empty
	}
	report("initialized trackers\n", print)
	// Stopwatch switch
	preTimeElapsed := time.Since(preTime)
	inTime := time.Now()

	// Choose algorithm to run
	var result []int
	switch optimMode {
	case "Lazy":
		result = lazyGreedy(graph, groupReqTracker, cardinality, closestInCoreset, print)
	case "Random":
		result = subSampleRange(graph.n, cardinality)
	default:
		result = []int{}
	}

	// Return result
	inTimeElapsed := time.Since(inTime)
	finalScore := computeScore(graph, result, closestInCoreset)
	return result, finalScore, preTimeElapsed, inTimeElapsed
}

func getGraph(dbName string, collectionName string, print bool) Graph {
	// Get collection
	collection := getMongoCollection(dbName, collectionName)
	n := getCollectionSize(collection)
	// Initialize results
	graph := Graph{
		affinityMatrix: make([][]float64, n),
		groups:         make([]int, n),
		n:              n,
	}
	for i := 0; i < n; i++ {
		graph.affinityMatrix[i] = make([]float64, n)
	}
	// Get cursor for all entries in collection
	cur := getFullCursor(collection)
	defer cur.Close(context.Background())
	// Fill out the graph
	for i := 0; cur.Next(context.Background()); i++ {
		point := getEntryFromCursor(cur)

		graph.affinityMatrix[i] = point.Affinities
		graph.groups[i] = point.Group
		report("loading db to memory "+strconv.Itoa(i)+"\r", print)
	}
	return graph
}

func getPartialGraph(dbName string, collectionName string, print bool, ssSize int, slices []int) Graph {

	collection := getMongoCollection(dbName, collectionName)

	graph := Graph{
		affinityMatrix: make([][]float64, ssSize),
		groups:         make([]int, ssSize),
		n:              ssSize,
	}

	cur := getSliceCursor(collection, slices)

	defer cur.Close(context.Background())

	for i := 0; cur.Next(context.Background()); i++ {
		point := getEntryFromCursor(cur)
		graph.affinityMatrix[i] = getSSAffinities(slices, point.Affinities, ssSize)

		report("point index: "+strconv.Itoa(point.Index)+"\n", print)
		report("Points affinities: "+listString(getSSAffinities(slices, point.Affinities, ssSize), ssSize), print)
		graph.groups[i] = point.Group
		report("loading db to memory "+strconv.Itoa(i)+"\r", print)
	}
	return graph
}

func listString(list []float64, size int) string {

	sList := make([]string, len(list))
	for i := 0; i < size; i++ {
		sList[i] = fmt.Sprintf("%.2f", list[i])

	}

	result := strings.Join(sList, ", ")
	return result
}
func getSSAffinities(slices []int, affinities []float64, ssSize int) []float64 {

	ssAffinities := make([]float64, ssSize)

	for i := 0; i < ssSize; i++ {
		ssAffinities[i] = affinities[slices[i]]
	}

	return ssAffinities
}

func marginalGain(graph Graph, index int, closestInCoreset []int) float64 {
	if closestInCoreset[0] < 0 { // Coreset is empty
		gain := 0.0
		for i := 0; i < graph.n; i++ { // New point is always closest
			print("i, j: " + strconv.Itoa(index) + ", " + strconv.Itoa(i) + "\n")
			gain += graph.getSim(index, i)
		}
		return gain
	} else {
		gain := 0.0
		for i := 0; i < graph.n; i++ {
			new_sim := graph.getSim(index, i)
			old_sim := graph.getSim(closestInCoreset[i], i)
			if new_sim > old_sim {
				gain += new_sim - old_sim
			}
		}
		return gain
	}
}

func updateTracker(graph Graph, index int, closestInCoreset []int, groupReqTracker []int) {
	for i := 0; i < graph.n; i++ {
		if closestInCoreset[i] < 0 {
			closestInCoreset[i] = index
		} else {
			new_sim := graph.getSim(index, i)
			old_sim := graph.getSim(closestInCoreset[i], i)
			if new_sim > old_sim {
				closestInCoreset[i] = index
			}
		}
	}
	group := graph.groups[index]
	groupReqTracker[group] = groupReqTracker[group] - 1
}

func computeScore(graph Graph, coreset []int, closestInCoreset []int) float64 {
	score := 0.0
	for i := 0; i < graph.n; i++ {
		if closestInCoreset[i] < 0 {
			best_sim := 0.0
			for j := 0; j < len(coreset); j++ {
				sim := graph.getSim(i, j)
				if sim > best_sim {
					best_sim = sim
				}
			}
			score += best_sim
		} else {
			score += graph.getSim(closestInCoreset[i], i)
		}
	}
	return score
}
