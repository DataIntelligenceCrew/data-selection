package main

import (
	"strconv"
)

func lazyLazyGreedy(graph Graph, coverageTracker []int, groupTracker []int,
	coreset []int, candidates map[int]bool, cardinality int, threads int,
	print bool, copyTrackers bool, eps float64) []int {
	// Initial report
	report("Executing lazylazy greedy algorithm...\n", print)

	// Copy trackers if necessary
	// (This is for if lazyGreedy is ran as a subroutine for GreeDi)
	var newCoverageTracker []int
	var newGroupTracker []int
	if copyTrackers {
		newCoverageTracker = make([]int, len(coverageTracker))
		newGroupTracker = make([]int, len(groupTracker))
		copy(newCoverageTracker, coverageTracker)
		copy(newGroupTracker, groupTracker)
	} else {
		newCoverageTracker = coverageTracker
		newGroupTracker = groupTracker
	}

	// Initialize sets & constants
	n := len(candidates)
	s := int(eps * float64(n))
	objScore := 0

	// Repeat main loop until all trackers are complete, or the candidate pool
	// is dried out, or cardinality constraint is met
	report("Entering the main loop...\n", print)

	for i := 0; (cardinality < 0 || len(coreset) < cardinality) && (sum(coverageTracker)+sum(groupTracker) >= objScore); i++ {
		// Take a subsample of the candidates
		sample := subSampleSet(candidates, s)
		splitSample := splitSet(sample, threads)

		// Creat a list of arguments to pass into each worker
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			arg := []interface{}{
				graph,
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
		decrementTrackers(graph, chosen.index, coverageTracker, groupTracker)
		delete(candidates, chosen.index)
		report("\rIteration "+strconv.Itoa(i)+" complete with marginal gain "+strconv.Itoa(chosen.gain)+", remaining candidates"+strconv.Itoa(len(candidates)), print)
	}
	report("\n", print)
	return coreset
}

func lazyLazyWorker(graph Graph, candidates map[int]bool, coverageTracker []int,
	groupTracker []int) *Result {
	// Iterate over points found by the query
	result := setEmptyResult()
	for i, _ := range candidates {
		gain := marginalGain(graph, i, coverageTracker, groupTracker)
		if gain > result.gain { // Update if better marginal gain found
			result.index = i
			result.gain = gain
		}
	}
	return result
}
