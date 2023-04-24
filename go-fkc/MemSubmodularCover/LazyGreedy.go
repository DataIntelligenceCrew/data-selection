package main

import (
	"container/heap"
	"strconv"
)

/*
Runs the LazyGreedy algorithm.
*/
func lazyGreedy(graph Graph, coverageTracker []int, groupTracker []int,
	coreset []int, candidates map[int]bool, cardinality int, threads int,
	print bool, copyTrackers bool) []int {
	// Initial report
	report("Executing lazy greedy algorithm...\n", print)
	report("remaining score: "+strconv.Itoa(remainingScore(coverageTracker, groupTracker)), print)

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

	// Initialize stuff
	n := len(candidates)

	// Compute initial marginal gains
	splitCandidates := splitSet(candidates, threads)
	args := make([][]interface{}, threads)
	for t := 0; t < threads; t++ {
		arg := []interface{}{
			graph,
			newCoverageTracker,
			newGroupTracker,
			splitCandidates[t],
		}
		args[t] = arg
	}
	initialGains := concurrentlyExecute(getMarginalGains, args)
	report("Initial marginal gains computed\n", print)

	// Initialize priority queue
	candidatesPQ := make(PriorityQueue, n)
	i := 0
	for result := range initialGains {
		if items, ok := result.([]*Item); ok {
			for _, item := range items {
				candidatesPQ[i] = item
				i++
			}
		}
	}
	heap.Init(&candidatesPQ)

	// Repeat main loop until all trackers are complete, or the candidate pool
	// is dried out, or cardinality constraint is met
	report("Entering the main loop...\n", print)
	for i := 0; sum(newCoverageTracker)+sum(newGroupTracker) > 0 && len(candidatesPQ) > 1 && (cardinality < 0 || len(coreset) < cardinality); i++ {
		for j := 1; true; j++ {
			// Get the next candidate point & its marginal gain
			index := heap.Pop(&candidatesPQ).(Item).value
			gain := marginalGain(graph, index, coverageTracker, groupTracker)

			// Optimal element found if it's the last possible option or
			// if its marginal gain is optimal
			if len(candidatesPQ) == 0 || gain >= PeekPriority(&candidatesPQ) {
				coreset = append(coreset, index)
				decrementTrackers(graph, index, coverageTracker, groupTracker)
				report("\rIteration "+strconv.Itoa(i)+" complete with marginal gain "+strconv.Itoa(gain)+", remaining candidates: "+strconv.Itoa(len(candidatesPQ))+", and elements reevaluated: "+strconv.Itoa(j), print)
				break // End search for next coreset point
			} else { // Add point back to heap with updated marginal gain
				item := &Item{
					value:    index,
					priority: gain,
				}
				heap.Push(&candidatesPQ, item)
			}
		}
	}
	report("\n", print)
	return coreset
}
