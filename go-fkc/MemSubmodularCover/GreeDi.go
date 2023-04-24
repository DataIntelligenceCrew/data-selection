package main

import (
	"fmt"
)

func greeDi(graph Graph, coverageTracker []int, groupTracker []int,
	coreset []int, candidates map[int]bool, cardinality int, threads int,
	print bool, optimMode int, eps float64) []int {
	fmt.Println("Executing GreeDi...")

	// Split candidates into subsets
	splitCandidates := splitSet(candidates, threads)

	// Call centralized greedy as goroutines with split candidates
	var results chan interface{}
	if optimMode == 0 { // LazyGreedy
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			arg := []interface{}{
				graph,
				coverageTracker,
				groupTracker,
				[]int{},
				splitCandidates[t],
				cardinality,
				1,
				false,
				true,
			}
			args[t] = arg
		}
		results = concurrentlyExecute(lazyGreedy, args)
	} else if optimMode == 1 { // LazyLazyGreedy
		args := make([][]interface{}, threads)
		for t := 0; t < threads; t++ {
			arg := []interface{}{
				graph,
				coverageTracker,
				groupTracker,
				[]int{},
				splitCandidates[t],
				cardinality,
				1,
				false,
				true,
				eps,
			}
			args[t] = arg
		}
		results = concurrentlyExecute(lazyLazyGreedy, args)
	}

	// Filtered candidates = union of solutions from each thread
	filteredCandidates := make(map[int]bool, cardinality*threads)
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
	fmt.Println("Executing centralized LazyGreedy...")
	return lazyGreedy(graph, coverageTracker, groupTracker, []int{}, filteredCandidates, cardinality, threads, false, false)
}
