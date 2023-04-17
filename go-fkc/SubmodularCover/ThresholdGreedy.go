package main

import (
	"container/heap"
	"strconv"

	"go.mongodb.org/mongo-driver/mongo"
)

func thresholdGreedy(collection *mongo.Collection, coverageTracker []int, groupTracker []int, candidates map[int]bool, threads int, print bool, eps float64) []int {
	report("Executing threshold greedy algorithm...\n", print)

	// Running bucket threshold
	// INTENTIONALLY BROKE MAXIMALGAIN
	threshold := float64(100) * (1.0 - eps) // Initial score threshold
	n := getCollectionSize(collection)
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
	report("Entering the main loop...\n", print)
	for i := 0; sum(coverageTracker)+sum(groupTracker) > 0 && len(candidatesPQ) > 1; i++ {
		item := heap.Pop(&candidatesPQ).(*Item)
		index := item.value
		point := getPointFromDB(collection, item.value)
		gain := marginalGain(point, coverageTracker, groupTracker)

		nextGain := PeekPriority(&candidatesPQ)
		// Add any candidate above the threshold
		if float64(gain) >= threshold || gain >= nextGain {
			coreset = append(coreset, index)
			decrementTrackers(&point, coverageTracker, groupTracker)
			report("\rIteration "+strconv.Itoa(i)+" complete with marginal gain "+strconv.Itoa(gain)+", remaining candidates: "+strconv.Itoa(len(candidatesPQ)), print)
			// Decrease threshold if no other point with gain above threshold exists
			for float64(nextGain) < threshold {
				threshold *= 1.0 - eps
			}
		} else {
			item := &Item{
				value:    index,
				priority: gain,
			}
			heap.Push(&candidatesPQ, item)
		}
	}
	report("\n", print)
	return coreset
}
