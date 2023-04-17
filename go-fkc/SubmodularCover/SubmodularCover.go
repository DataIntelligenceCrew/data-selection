package main

import (
	"context"
	"fmt"

	"go.mongodb.org/mongo-driver/mongo"
)

/*
*
Optimization modes:
0: Classic greedy
1: Lazy greedy
2: Lazy Lazy greedy
3: Multilevel with lazylazy -> lazy
2: Distributed submodular cover (DisCover) using GreeDi & lazygreedy as subroutines
*/
func SubmodularCover(dbName string, collectionName string, coverageReq int,
	groupReqs []int, optimMode int, threads int, dense bool, eps float64, objRatio float64, print bool) []int {
	// Get the collection from DB
	collection := getMongoCollection(dbName, collectionName)
	report("obtained collection\n", true)

	// Initialize trackers
	n := getCollectionSize(collection)
	coverageTracker, groupReqs, coreset := getTrackers(collection, coverageReq, groupReqs, dense, n)
	fmt.Println(coverageTracker, len(coverageTracker))
	decrementAllOrderedTrackers(collection, coreset, coverageTracker, groupReqs)
	fmt.Println(coverageTracker, len(coverageTracker))
	//maximalGain += groupReqs[0] // Largest initial gain
	report("initialized trackers\n", true)

	// Choose algorithm to run
	switch optimMode {
	case 0:
		candidates := setMinus(rangeSet(n), sliceToSet(coreset))
		result := classicGreedy(collection, coverageTracker, groupReqs, coreset, candidates, -1, threads, print)
		return result
	case 1:
		candidates := setMinus(rangeSet(n), sliceToSet(coreset))
		result := lazyGreedy(collection, coverageTracker, groupReqs, coreset, candidates, -1, threads, print, false)
		return result
	case 2:
		result := disCover(collection, coreset, coverageTracker, groupReqs, threads, 0.5, print, coverageReq, n, dense)
		return result
	default:
		return []int{}
	}
}

func getTrackers(collection *mongo.Collection, coverageReq int, groupReqs []int, dense bool, n int) ([]int, []int, []int) {
	if dense {
		coverageTracker := make([]int, n)
		for i := 0; i < n; i++ {
			coverageTracker[i] = coverageReq
		}
		//fmt.Println(len(coverageTracker))
		return coverageTracker, groupReqs, []int{}
	} else {
		coverageTracker := make([]int, n)
		coreset := make([]int, 0)
		cur := getFullCursor(collection)
		defer cur.Close(context.Background())
		for i := 0; cur.Next(context.Background()); i++ {
			point := getEntryFromCursor(cur)
			numNeighbors := len(point.Neighbors)
			coverageTracker[i] = min(numNeighbors, coverageReq)
			group := point.Group
			//thisCoverageReq := min(numNeighbors, coverageReq)
			if numNeighbors <= coverageReq { // Add to coreset automatically
				coreset = append(coreset, point.Index)
				groupReqs[group] = max(0, groupReqs[group] - 1)
			}
			fmt.Printf("\rCoverage tracker iteration %d", i)
		}
		fmt.Printf("\n")
		return coverageTracker, groupReqs, coreset
	}
}

func marginalGain(point Point, coverageTracker []int, groupTracker []int) int {
	gain := 0
	for neighbor, _ := range point.Neighbors {
		gain += coverageTracker[neighbor]
	}
	gain += groupTracker[point.Group] // Marginal gain from group requirement
	return gain
}

func getMarginalGains(collection *mongo.Collection, coverageTracker []int,
	groupTracker []int, candidates map[int]bool) []*Item {
	// Query the database
	cur := getSetCursor(collection, candidates)
	defer cur.Close(context.Background())

	// Get results by iterating the cursor
	results := make([]*Item, 0)
	for cur.Next(context.Background()) {
		point := getEntryFromCursor(cur)
		gain := marginalGain(point, coverageTracker, groupTracker)
		item := &Item{
			value:    point.Index,
			priority: gain,
		}
		results = append(results, item)
	}
	return results
}

func notSatisfied(coverageTracker []int, groupTracker []int) bool {
	return sum(coverageTracker)+sum(groupTracker) > 0
}

func decrementTrackers(point *Point, coverageTracker []int, groupTracker []int) {
	for neighbor := range point.Neighbors {
		coverageTracker[neighbor] = max(0, coverageTracker[neighbor]-1)
	}
	group := point.Group
	groupTracker[group] = max(0, groupTracker[group]-1)
}

func decrementAllOrderedTrackers(collection *mongo.Collection, points []int, coverageTracker []int, groupTracker []int) {
	cur := getFullCursor(collection)
	defer cur.Close(context.Background())

	i := 0
	for cur.Next(context.Background()) {
		point := getEntryFromCursor(cur)
		if points[i] == point.Index {
			decrementTrackers(&point, coverageTracker, groupTracker)
			i++
		}
		fmt.Print("%d\r", i)
	}
	fmt.Println()
}

func decrementAllTrackers(collection *mongo.Collection, points []int, coverageTracker []int, groupTracker []int) {
	for i := 0; i < len(points); i++ {
		point := getPointFromDB(collection, points[i])
		decrementTrackers(&point, coverageTracker, groupTracker)
		fmt.Print("%d\r", i)
	}
	fmt.Println()
}

func remainingScore(coverageTracker []int, groupTracker []int) int {
	return sum(coverageTracker) + sum(groupTracker)
}

// Return all points whose coverage is below k
func allBelowCovThreshold(collection *mongo.Collection, threads int, k int, n int) []int {
	chunkSize := n / threads
	args := make([][]interface{}, threads)
	for t := 0; t < threads; t++ {
		lo := t * chunkSize
		hi := lo + chunkSize - 1
		arg := []interface{}{
			collection,
			lo,
			hi,
			k,
		}
		args[t] = arg
	}
	results := concurrentlyExecute(belowCovTreshold, args)
	ret := make([]int, 0)
	for r := range results {
		if res, ok := r.([]int); ok {
			for i := 0; i < len(res); i++ {
				ret = append(ret, res[i])
			}
		} else {
			fmt.Println("Interpret error")
		}
	}
	return ret
}

func belowCovTreshold(collection *mongo.Collection, lo int, hi int, k int) []int {
	ret := make([]int, 0)
	cur := getRangeCursor(collection, lo, hi)
	defer cur.Close(context.Background())

	for cur.Next(context.Background()) {
		point := getEntryFromCursor(cur)
		if len(point.Neighbors) < k {
			ret = append(ret, point.Index)
		}
	}
	return ret
}
