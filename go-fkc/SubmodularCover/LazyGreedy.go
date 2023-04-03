package main

import (
	"container/heap"
	"fmt"
	"strconv"

	"go.mongodb.org/mongo-driver/mongo"
)

/**
Runs the
*/

func lazyGreedy(collection *mongo.Collection, coverageTracker []int,
	groupTracker []int, candidates map[int]bool, constraint int, threads int,
	print bool) []int {
	report("Executing lazy greedy algorithm...\n", print)
	fmt.Println("remaining score: ", remainingScore(coverageTracker, groupTracker))

	// Initialize sets
	n := len(candidates)
	coreset := make([]int, 0)

	// Compute initial marginal gains
	splitCandidates := splitSet(candidates, threads)
	args := make([][]interface{}, threads)
	for t := 0; t < threads; t++ {
		arg := []interface{}{
			collection,
			coverageTracker,
			groupTracker,
			splitCandidates[t],
		}
		args[t] = arg
	}
	initialGains := concurrentlyExecute(getMarginalGains, args)

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
	for i := 0; sum(coverageTracker)+sum(groupTracker) > 0 && len(candidatesPQ) > 0 && (constraint < 0 || len(coreset) < constraint); i++ {
		for j := 1; true; j++ {
			// Get the next candidate point & its marginal gain
			index := heap.Pop(&candidatesPQ).(*Item).value
			point := getPointFromDB(collection, index)
			gain := marginalGain(point, coverageTracker, groupTracker, threads)

			// Optimal element found if it's the last possible option or
			// if its marginal gain is optimal
			if len(candidatesPQ) == 0 || gain >= PeekPriority(&candidatesPQ) {
				coreset = append(coreset, index)
				decrementTrackers(&point, coverageTracker, groupTracker)
				report("\rIteration "+strconv.Itoa(i)+" complete with marginal gain "+strconv.Itoa(gain)+", remaining candidates: "+strconv.Itoa(len(candidatesPQ))+", and elements reevaluated: "+strconv.Itoa(j), print)
				break // End search
			} else { // Add the point back to heap with updated marginal gain
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
