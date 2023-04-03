package main

import (
	"container/heap"
	"fmt"
	"log"
	"math/rand"
	"reflect"
	"sync"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

/**
Utility related to reasoning about the result of a marginal gain evaluation
*/

type Result struct {
	index int
	gain  int
}

func setEmptyResult() *Result {
	return &Result{
		index: -1,
		gain:  -1,
	}
}

func getBestResult(results chan interface{}) *Result {
	best := setEmptyResult()
	for r := range results {
		if res, ok := r.(*Result); ok {
			if res.gain > best.gain {
				best = res
			}
		} else {
			fmt.Println("Interpret error")
		}
	}
	return best
}

/**
The type of a database entry.
*/

type Point struct {
	ID        primitive.ObjectID `bson:"_id"`
	Index     int                `bson:"index"`
	Group     int                `bson:"group"`
	Neighbors []bool             `bson:"neighbors"`
}

/**
Miscellaneous functions.
*/

// Very basic error handling
func handleError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func removeFromSlice(s []int, index int) []int {
	for i := 0; i < len(s); i++ {
		if i == index {
			s[i] = s[len(s)-1]
			return s[:len(s)-1]
		}
	}
	return s
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func sum(slice []int) int {
	sum := 0
	for i := range slice {
		sum += slice[i]
	}
	return sum
}

func rangeSlice(n int) []int {
	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = i
	}
	return result
}

func rangeSet(n int) map[int]bool {
	result := make(map[int]bool, n)
	for i := 0; i < n; i++ {
		result[i] = true
	}
	return result
}

func deleteAllFromSet(set map[int]bool, keys []int) map[int]bool {
	for i := 0; i < len(keys); i++ {
		key := keys[i]
		delete(set, key)
	}
	return set
}

func report(message string, print bool) {
	if print {
		fmt.Printf(message)
	}
}

func mapToSlice(set map[int]bool) []int {
	keys := make([]int, 0)
	for k := range set {
		keys = append(keys, k)
	}
	return keys
}

func splitSet(set map[int]bool, threads int) []map[int]bool {
	result := make([]map[int]bool, threads)
	countPerSplit := (len(set) / threads) + 1
	for i := 0; i < threads; i++ {
		result[i] = make(map[int]bool, 0)
	}
	i := 0
	for key := range set {
		assign := i / countPerSplit
		result[assign][key] = true
		i++
	}
	return result
}

func subSampleSet(set map[int]bool, size int) map[int]bool {
	result := make(map[int]bool, size)
	i := 0
	for item := range set {
		prob := float64(size-len(result)) / float64(len(set)-i)
		if rand.Float64() <= prob { // Success
			result[item] = true
		}
		i++
	}
	return result
}

func notSatisfiedIndices(coverageTracker []int) []int {
	result := make([]int, 0)
	for i := 0; i < len(coverageTracker); i++ {
		if coverageTracker[i] > 0 {
			result = append(result, i)
		}
	}
	return result
}

func sliceToSet(slice []int) map[int]bool {
	result := make(map[int]bool, len(slice))
	for i := 0; i < len(slice); i++ {
		result[slice[i]] = true
	}
	return result
}

func setMinus(foo map[int]bool, bar map[int]bool) map[int]bool {
	result := make(map[int]bool)
	for key, _ := range foo {
		if !bar[key] {
			result[key] = true
		}
	}
	return result
}

/**
Everything required to implement priority queue.
*/

type Item struct {
	value    int
	priority int
	index    int
}

func getEmptyItem() *Item {
	return &Item{
		value:    -1,
		priority: -1,
		index:    0,
	}
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x any) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // avoid memory leak
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// update modifies the priority and value of an Item in the queue.
func (pq *PriorityQueue) update(item *Item, value int, priority int) {
	item.value = value
	item.priority = priority
	heap.Fix(pq, item.index)
}

func PeekPriority(pq *PriorityQueue) int {
	//return (*pq)[len(*pq)-1].priority
	return (*pq)[0].priority
}

func removeFromPQ(pq *PriorityQueue, pos int) {
	pq.Swap(pos, len(*pq)-1)
	pq.Pop()
}

/**
Concurrency wrapper to avoid rewriting the same boilerplate code
*/

func concurrentlyExecute(f interface{}, args [][]interface{}) chan interface{} {
	threads := len(args)
	results := make(chan interface{}, threads)
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func(args []interface{}) {
			argValues := make([]reflect.Value, len(args))
			for i, arg := range args {
				argValues[i] = reflect.ValueOf(arg)
			}
			result := reflect.ValueOf(f).Call(argValues)[0].Interface()
			results <- result
			wg.Done()
		}(args[t])
	}
	wg.Wait()
	close(results)
	return results
}
