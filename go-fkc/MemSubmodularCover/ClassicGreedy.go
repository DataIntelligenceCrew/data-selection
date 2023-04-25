package main

import (
	"strconv"
)

/*
Runs the classic greedy algorithm for submodular cover on the given candidates
pool until all trackers are zeroed out.
*/
func classicGreedy(graph Graph, coverageTracker []int, groupTracker []int,
	coreset []int, candidates map[int]bool, cardinality int, threads int, print bool) []int {
	report("Executing classic greedy algorithm...\n", print)

	// Initialize stuff
	chunkSize := len(graph.adjMatrix) / threads

	// Repeat main loop until all requirements are met or candidate pool
	// is dried out, or cardinality constraint is met
	report("Entering the main loop...\n", print)
	for notSatisfied(coverageTracker, groupTracker) && len(candidates) > 0 && (cardinality < 0 || len(coreset) < cardinality) {
		// Creat a list of arguments to pass into each worker
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			lo := t * chunkSize
			hi := lo + chunkSize - 1
			arg := []interface{}{
				graph,
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
		bestResult := getBestResult(results)
		chosen := bestResult.index
		coreset = append(coreset, chosen)
		delete(candidates, chosen) // Remove chosen point from candidates pool
		// Update trackers
		decrementTrackers(graph, chosen, coverageTracker, groupTracker)
		report("\rIteration: "+strconv.Itoa(len(coreset))+" complete with marginal gain "+strconv.Itoa(bestResult.gain), print)
	}
	report("\n", print)
	return coreset
}

func classicWorker(graph Graph, candidates map[int]bool, coverageTracker []int,
	groupTracker []int, lo int, hi int) *Result {
	// Iterate over points found by the query
	result := setEmptyResult()
	for i := lo; i <= hi; i++ {
		if candidates[i] {
			gain := marginalGain(graph, i, coverageTracker, groupTracker)
			if gain > result.gain {
				result.index = i
				result.gain = gain
			}
		}
	}
	return result
}
