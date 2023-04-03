package main

import (
	"context"
	"fmt"
	"strconv"

	"go.mongodb.org/mongo-driver/mongo"
)

/**
Runs the classic greedy algorithm for submodular cover on the given candidates
pool until all trackers are zeroed out.
*/

func classicGreedy(collection *mongo.Collection, coverageTracker []int,
	groupTracker []int, candidates map[int]bool, constraint int, threads int,
	print bool) []int {
	report("Executing classic greedy algorithm...\n", print)

	// Initialize sets
	n := getCollectionSize(collection)
	coreset := make([]int, 0)
	chunkSize := n / threads

	// Repeat main loop until all requirements are met or candidate pool
	// is dried out, or cardinality constraint is met
	report("Entering the main loop...\n", print)
	for notSatisfied(coverageTracker, groupTracker) && len(candidates) > 0 && (constraint < 0 || len(coreset) < constraint) {
		// Creat a list of arguments to pass into each worker
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			lo := t * chunkSize
			hi := lo + chunkSize - 1
			arg := []interface{}{
				collection,
				candidates,
				coverageTracker,
				groupTracker,
				lo,
				hi,
			}
			args[t] = arg
		}
		// Actual work of concurrent candidate evaluation
		results := concurrentlyExecute(classicWorker, args)

		// End-of-iteration bookkeeping
		chosen := getBestResult(results)
		coreset = append(coreset, chosen.index)
		delete(candidates, chosen.index)
		point := getPointFromDB(collection, chosen.index)
		decrementTrackers(&point, coverageTracker, groupTracker)
		report("\rIteration: "+strconv.Itoa(len(coreset))+" complete with marginal gain "+strconv.Itoa(chosen.gain), print)
		if chosen.gain == 0 {
			fmt.Printf("%v %v\n", coverageTracker, groupTracker)
		}
	}
	report("\n", print)
	return coreset
}

func classicWorker(collection *mongo.Collection, candidates map[int]bool, coverageTracker []int,
	groupTracker []int, lo int, hi int) *Result {
	// Query the points in range lo...hi
	cur := getRangeCursor(collection, lo, hi)
	defer cur.Close(context.Background())

	// Iterate over points found by the query
	result := setEmptyResult()
	for cur.Next(context.Background()) { // Iterate over query results
		point := getEntryFromCursor(cur)
		// If the point is a candidate AND it is assigned to this worker thread
		if candidates[point.Index] {
			gain := marginalGain(point, coverageTracker, groupTracker, 1)
			if gain > result.gain { // Update if better marginal gain found
				result.index = point.Index
				result.gain = gain
			}
		}
	}
	return result
}
