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
	coverageTracker, maximalGain := getCoverageTracker(collection, coverageReq, dense, n)
	maximalGain += groupReqs[0] // Largest initial gain
	report("initialized trackers\n", true)

	// Choose algorithm to run
	switch optimMode {
	case 0:
		result := classicGreedy(collection, coverageTracker, groupReqs, rangeSet(n), -1, threads, print)
		return result
	case 1:
		result := lazyGreedy(collection, coverageTracker, groupReqs, rangeSet(n), -1, threads, print)
		return result
	case 2:
		result := lazyLazyGreedy(collection, coverageTracker, groupReqs, rangeSet(n), -1, threads, print, eps, 1.0)
		return result
	case 3:
		firstStage := lazyLazyGreedy(collection, coverageTracker, groupReqs, rangeSet(n), -1, threads, print, eps, objRatio)
		candidates := setMinus(rangeSet(n), sliceToSet(firstStage))
		secondStage := lazyGreedy(collection, coverageTracker, groupReqs, candidates, -1, threads, print)
		totalSolution := append(firstStage, secondStage...)
		return totalSolution
	case 4:
		result := disCover(collection, coverageTracker, groupReqs, threads, 0.2, print, coverageReq, n)
		return result
	case 5:
		result := thresholdGreedy(collection, coverageTracker, groupReqs, rangeSet(n), threads, print, eps, maximalGain)
		return result
	default:
		return []int{}
	}
}

func getCoverageTracker(collection *mongo.Collection, coverageReq int, dense bool, n int) ([]int, int) {
	if dense {
		coverageTracker := make([]int, n)
		for i := 0; i < n; i++ {
			coverageTracker[i] = coverageReq
		}
		//fmt.Println(len(coverageTracker))
		return coverageTracker, coverageReq * coverageReq
	} else {
		maxGain := 0
		coverageTracker := make([]int, 0)
		cur := getFullCursor(collection)
		defer cur.Close(context.Background())
		for i := 0; cur.Next(context.Background()); i++ {
			point := getEntryFromCursor(cur)
			numNeighbors := 0
			for i := 0; i < len(point.Neighbors); i++ {
				if point.Neighbors[i] {
					numNeighbors++
				}
			}
			thisCoverageReq := min(numNeighbors, coverageReq)
			coverageTracker = append(coverageTracker, thisCoverageReq)
			maxGain = max(maxGain, thisCoverageReq)
			fmt.Printf("\rCoverage tracker iteration %d", i)
		}
		fmt.Printf("\n")
		return coverageTracker, maxGain
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
	for i := 0; i < len(point.Neighbors); i++ {
		if point.Neighbors[i] {
			coverageTracker[i] = max(0, coverageTracker[i]-1)
		}
	}
	gr := point.Group
	val := groupTracker[gr]
	groupTracker[gr] = max(0, val-1)
}

func decrementAllTrackers(collection *mongo.Collection, points []int, coverageTracker []int, groupTracker []int) {
	for i := 0; i < len(points); i++ {
		point := getPointFromDB(collection, points[i])
		decrementTrackers(&point, coverageTracker, groupTracker)
	}
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
