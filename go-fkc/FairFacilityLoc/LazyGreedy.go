package main

import (
	"container/heap"
	"strconv"
)

/*
Runs the LazyGreedy algorithm.
Client syntax:
result = lazyGreedy(graph, groupReqTracker, cardinality, closestInCoreset, print)
*/
func lazyGreedy(graph Graph, groupReqTracker []int, cardinality int,
		closestInCoreset []int, print bool) []int {
	// Initial report
	report("Executing lazy greedy algorithm...\n", print)

	// Initialize PQ with initial marginal gains
	candidatesPQ := make(PriorityQueue, graph.n)
	for i := 0; i < graph.n; i++ {
		gain := marginalGain(graph, i, closestInCoreset)
		item := &Item{
			value: i,
			priority: gain,
			index: i,
		}
		candidatesPQ[i] = item
	}
	heap.Init(&candidatesPQ)

	// Initialize coreset
	coreset := make([]int, 0)

	// Repeat main loop until cardinality constraint is met
	report("Entering the main loop...\n", print)
	for i := 0; i < cardinality; i++ {
		// Evaluate elements at top of PQ
		for j := 1; true; j++ {
			// Get the next candidate point & its marginal gain
			index := heap.Pop(&candidatesPQ).(*Item).value
			gain := marginalGain(graph, index, closestInCoreset)
			group := graph.groups[index]

			if groupReqTracker[group] <= 0 { // Group cnt requirement was met
				continue // Simply drop this point
			} else {
				// Optimal element found if it's the last possible option or
				// if its marginal gain is optimal
				if len(candidatesPQ) == 0 || gain >= PeekPriority(&candidatesPQ) {
					coreset = append(coreset, index)
					updateTracker(graph, index, closestInCoreset, groupReqTracker)
					// Report this iteration
					report_str := "\rIteration "+strconv.Itoa(i)+" complete with "
					report_str += "marginal gain "+strconv.FormatFloat(gain, 'f', 6, 64)
					report_str += ", and elements reevaluated: " + strconv.Itoa(j)
					report_str += ", coreset size: " + strconv.Itoa(len(coreset))
					report_str += ", PQ size: " + strconv.Itoa(len(candidatesPQ))
					report(report_str, print)
					break
				} else { // Push suboptimal element back with updated gain
					item := &Item{
						value: index,
						priority: gain,
					}
					heap.Push(&candidatesPQ, item)
				}
			}
		}
	}
	report("\n", print)
	return coreset
}
