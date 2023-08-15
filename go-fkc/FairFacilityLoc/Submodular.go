package main

import (
	"context"
	"strconv"
	"time"
)

type Graph struct {
	affinityMatrix [][]float64
	groups         []int
	n              int
}

/*
Client syntax:
coreset, funcVal, preTime, inTime := SubmodularCover(db, collection, groupReq, 
	groupCnt, optim, threads, cardinality, iterPrint)
*/
func SubmodularCover(dbName string, collectionName string, groupReq int, 
					groupCnt int, optimMode string, threads int, cardinality int,
					print bool) ([]int, float64, time.Duration, time.Duration) {
	preTime := time.Now()
	// Import & Initialize all stuff
	graph := getGraph(dbName, collectionName, print)
	groupReqTracker := make([]int, groupCnt) // Remaining group cnt reqs
	for i := range groupReqTracker { // Initialize with groupReq
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
		groups: make([]int, n),
		n: n,
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

func marginalGain(graph Graph, index int, closestInCoreset []int) float64 {
	if closestInCoreset[0] < 0 { // Coreset is empty
		gain := 0.0
		for i := 0; i < graph.n; i++ { // New point is always closest
			gain += graph.affinityMatrix[index][i]
		}
		return gain
	} else {
		gain := 0.0
		for i := 0; i < graph.n; i++ {
			new_sim := graph.affinityMatrix[index][i]
			old_sim := graph.affinityMatrix[closestInCoreset[i]][i]
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
			new_sim := graph.affinityMatrix[index][i]
			old_sim := graph.affinityMatrix[closestInCoreset[i]][i]
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
				sim := graph.affinityMatrix[i][j]
				if sim > best_sim {
					best_sim = sim
				}
			}
			score += best_sim
		} else {
			score += graph.affinityMatrix[closestInCoreset[i]][i]
		}
	}
	return score
}
