package main

import (
	"context"
	"strconv"

	"go.mongodb.org/mongo-driver/mongo"
)

func lazyLazyGreedy(collection *mongo.Collection, coverageTracker []int,
	groupTracker []int, candidates map[int]bool, constraint int, threads int,
	print bool, eps float64, objRatio float64) []int {
	report("Executing lazylazy greedy algorithm...\n", print)

	// Initialize sets & constants
	n := len(candidates)
	if constraint < 0 { // If cardinality constraint flag was set to -1, then
		constraint = n // cardinality constraint is equal to n (i.e. no bound)
	}
	s := int(eps * float64(n))
	coreset := make([]int, 0)
	initialObj := sum(coverageTracker) + sum(groupTracker)
	objScore := int((1 - objRatio) * float64(initialObj))

	// Repeat main loop until all trackers are complete, or the candidate pool
	// is dried out, or cardinality constraint is met
	report("Entering the main loop...\n", print)

	for i := 0; (len(coreset) < constraint) && (sum(coverageTracker)+sum(groupTracker) >= objScore); i++ {
		// Take a subsample of the candidates
		sample := subSampleSet(candidates, s)
		splitSample := splitSet(sample, threads)

		// Creat a list of arguments to pass into each worker
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			arg := []interface{}{
				collection,
				splitSample[t],
				coverageTracker,
				groupTracker,
			}
			args[t] = arg
		}
		// Actual work of concurrent candidate evaluation
		results := concurrentlyExecute(lazyLazyWorker, args)
		chosen := getBestResult(results)

		// Bookkeeping
		coreset = append(coreset, chosen.index)
		point := getPointFromDB(collection, chosen.index)
		decrementTrackers(&point, coverageTracker, groupTracker)
		delete(candidates, chosen.index)
		report("\rIteration "+strconv.Itoa(i)+" complete with marginal gain "+strconv.Itoa(chosen.gain)+", remaining candidates"+strconv.Itoa(len(candidates)), print)
	}
	report("\n", print)
	return coreset
}

func lazyLazyWorker(collection *mongo.Collection, candidates map[int]bool, coverageTracker []int,
	groupTracker []int) *Result {
	// Query the points in range lo...hi
	cur := getSetCursor(collection, candidates)
	defer cur.Close(context.Background())

	// Iterate over points found by the query
	result := setEmptyResult()
	for cur.Next(context.Background()) { // Iterate over query results
		point := getEntryFromCursor(cur)
		// If the point is a candidate AND it is assigned to this worker thread
		gain := marginalGain(point, coverageTracker, groupTracker, 1)
		if gain > result.gain { // Update if better marginal gain found
			result.index = point.Index
			result.gain = gain
		}
	}
	return result
}
