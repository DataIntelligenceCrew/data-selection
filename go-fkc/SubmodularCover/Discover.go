package main

import (
	"fmt"
	"math"
	"strconv"

	"go.mongodb.org/mongo-driver/mongo"
)

func disCover(collection *mongo.Collection, coverageTracker []int,
	groupTracker []int, threads int, alpha float64, print bool) []int {
	fmt.Println("Executing DisCover...")
	coreset := make([]int, 0)
	n := getCollectionSize(collection)
	candidates := make(map[int]bool) // Using map as a hashset
	for i := 0; i < n; i++ {         // Initial points
		candidates[i] = true
	}
	lambda := 1.0 / math.Sqrt(float64(threads))

	// Main logic loop
	fmt.Println("Entering the main loop...")
	cardinalityConstraint := 2
	for r := 1; notSatisfied(coverageTracker, groupTracker); r++ {
		// Run DisCover subroutine
		remainingBefore := sum(coverageTracker) + sum(groupTracker)
		newSet := greeDi(candidates, coverageTracker, groupTracker, threads, cardinalityConstraint, collection)
		coreset = append(coreset, newSet...)
		candidates = deleteAllFromSet(candidates, newSet)
		remainingAfter := sum(coverageTracker) + sum(groupTracker)
		// Decide whether to double cardinality coustraint or not
		if float64(remainingBefore-remainingAfter) < alpha*lambda*float64(remainingBefore) {
			cardinalityConstraint *= 2 // Double if marginal gain is too small
		}

		report("\rRound: "+strconv.Itoa(r)+", remaining candidates: "+strconv.Itoa(len(candidates)), print)
	}
	fmt.Printf("\n")
	return coreset
}

func greeDi(candidates map[int]bool, coverageTracker []int, groupTracker []int,
	threads int, cardinalityConstraint int, collection *mongo.Collection) []int {
	// Make a copy of trackers since we don't want to mess with them
	newCoverageTracker := make([]int, len(coverageTracker))
	newGroupTracker := make([]int, len(groupTracker))
	copy(newCoverageTracker, coverageTracker)
	copy(newGroupTracker, groupTracker)

	// Split candidates into subsets
	splitCandidates := splitSet(candidates, threads)

	// Call centralized greedy as goroutines with split candidates
	args := make([][]interface{}, threads)
	for t := 0; t < threads; t++ {
		arg := []interface{}{
			collection,
			newCoverageTracker,
			newGroupTracker,
			splitCandidates[t],
			cardinalityConstraint,
			1,
			false,
		}
		args[t] = arg
	}

	results := concurrentlyExecute(lazyGreedy, args)

	// Filtered candidates = union of solutions from each thread
	filteredCandidates := make(map[int]bool, cardinalityConstraint*threads)
	for r := range results {
		if res, ok := r.([]int); ok {
			for i := 0; i < len(res); i++ {
				filteredCandidates[res[i]] = true
			}
		} else {
			fmt.Println("Interpret error")
		}
	}

	// Run centralized greedy on the filtered candidates
	return lazyGreedy(collection, coverageTracker, groupTracker, filteredCandidates, cardinalityConstraint, threads, false)
}
