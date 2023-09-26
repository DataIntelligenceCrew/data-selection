package main

import (
	"context"
	"strconv"
	"time"
	"math"
)

type Graph struct {
	affinityMatrix [][]float64
	groups         []int
	groupCnts []int
	n              int
}

/*
Client syntax:
coreset, funcVal, preTime, inTime := Saturate(db, collection, 
	groupCnt, optim, threads, cardinality, iterPrint, alpha)
*/
func Saturate(dbName string, collectionName string, groupCnt int, optimMode string, 
				threads int, cardinality int, print bool, alpha float64) ([]int, 
				[]float64, time.Duration, time.Duration, int) {
	preTime := time.Now()
	// Import & Initialize all stuff
	// Graph contains affinityMatrix float[][], groups []int, groupCnts []int, and n int
	graph := getGraph(dbName, collectionName, groupCnt, print)

	// Function F_g for each group is the mean of similarities b/w each pt in a
	// group and its closest medoid
	c_min := 0.0 // Lower bound on c
	c_max := 1.0 // Upper bound on c

	// Initial best coreset candidate is the empty coreset
	a_best := []int{}
	a_best_closest := make([]int, graph.n)

	// The cardinality constraint for each subroutine is alpha * cardinality
	inner_cardinality := int(alpha * float64(cardinality))

	/*closestInCoreset := make([]int, graph.n) // Index of nearest point in coreset
	for i := 0; i < graph.n; i++ {
		closestInCoreset[i] = -1 // -1 means coreset is empty
	}*/
	report("initialized trackers\n", print)
	// Stopwatch switch
	preTimeElapsed := time.Since(preTime)
	inTime := time.Now()

	// The main while loop for binary search
	i := 0
	for ; c_max - c_min >= 1.0 / float64(groupCnt); i++ {
		c := (c_min + c_max) / 2.0 // Binary search the middle c
		a_hat, covered, closestInCoreset := GPC(graph, inner_cardinality, c, print) // Greedy Partial Cover Subroutine
		if !covered {
			c_max = c
		} else {
			c_min = c
			a_best = a_hat
			a_best_closest = closestInCoreset
		}
	}

	// Return result
	inTimeElapsed := time.Since(inTime)
	finalScores := computeScores(graph, a_best, a_best_closest)
	return a_best, finalScores, preTimeElapsed, inTimeElapsed, i
}

func getGraph(dbName string, collectionName string, groupCnt int, print bool) Graph {
	// Get collection
	collection := getMongoCollection(dbName, collectionName)
	n := getCollectionSize(collection)
	// Initialize results
	graph := Graph{
		affinityMatrix: make([][]float64, n),
		groups: make([]int, n),
		groupCnts: make([]int, groupCnt),
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
		graph.groupCnts[point.Group]++
		report("loading db to memory "+strconv.Itoa(i)+"\r", print)
	}
	return graph
}

// Function F^bar is the average across all groups, avg of min(average similarity 
// b/w a pt and its medoid, c)
// To compute marginal gain per group, sum marginal gain in the min function
// Then per group, divide the sum by the size of the group
// Finally, take the average of the per group function values
func marginalGain(graph Graph, index int, closestInCoreset []int, c float64) float64 {
	// Compute sum of marginal similarities per group
	marginal_gains_per_group := make([]float64, len(graph.groupCnts))
	for i := 0; i < graph.n; i++ {
		g := graph.groups[i]
		new_sim := graph.affinityMatrix[index][i]
		old_sim := 0.0
		if closestInCoreset[i] >= 0 {
			old_sim = graph.affinityMatrix[closestInCoreset[i]][i]
		}
		if new_sim > old_sim {
			marginal_gains_per_group[g] += new_sim - old_sim
		}
	}
	// Normalize to get average
	for g := 0; g < len(marginal_gains_per_group); g++ {
		marginal_gains_per_group[g] /= float64(graph.groupCnts[g])
	}
	// Take minimum b/w average and c per group
	for g := 0; g < len(marginal_gains_per_group); g++ {
		marginal_gains_per_group[g] = math.Min(marginal_gains_per_group[g], c)
	}
	// Find the average of the array
	sum_all_groups := 0.0
	for g := 0; g < len(marginal_gains_per_group); g++ {
		sum_all_groups += marginal_gains_per_group[g]
	}
	return sum_all_groups / float64(len(marginal_gains_per_group))
}

func updateTracker(graph Graph, index int, closestInCoreset []int) {
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
}

func computeScores(graph Graph, coreset []int, closestInCoreset []int) []float64 {
	sum_per_group := make([]float64, len(graph.groupCnts))
	for i := 0; i < graph.n; i++ {
		sum_per_group[graph.groups[i]] += graph.affinityMatrix[closestInCoreset[i]][i]
	}
	for g := 0; g < len(sum_per_group); g++ {
		sum_per_group[g] /= float64(graph.groupCnts[g])
	}
	return sum_per_group
}
