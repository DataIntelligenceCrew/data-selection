package main

import (
	"flag"
	"fmt"
	"time"
)

func main() {

	// Define command-line flags
	dbFlag := flag.String("db", "dummydb", "MongoDB DB")
	collectionFlag := flag.String("col", "n1000d3m5r20", "ollection containing points")
	coverageFlag := flag.Int("k", 20, "k-coverage requirement")
	groupReqFlag := flag.Int("g", 100, "group count requirement")
	groupCntFlag := flag.Int("m", 5, "number of groups")
	optimFlag := flag.Int("optim", 0, "optimization mode")
	threadsFlag := flag.Int("t", 1, "number of threads")
	dense := flag.Bool("dense", true, "whether the graph is denser than the k-Coverage requirement")
	eps := flag.Float64("eps", 0.1, "portion of dataset randomly sampled in each iteration of LazyLazy")
	objRatio := flag.Float64("objratio", 0.9, "portion of objective function to be satisfied with LazyLazy before switching to Lazy")
	iterPrint := flag.Bool("iterprint", true, "whether to report each iteration's progress")
	//batchSize := flag.Int("batch", 10000, "number of entries to query from MongoDB at once")

	// Parse all flags
	flag.Parse()

	// Make the groupReqs array
	groupReqs := make([]int, *groupCntFlag)
	for i := 0; i < *groupCntFlag; i++ {
		groupReqs[i] = *groupReqFlag
	}

	// Run submodularCover
	start := time.Now()
	result := SubmodularCover(*dbFlag, *collectionFlag, *coverageFlag, groupReqs, *optimFlag, *threadsFlag, *dense, *eps, *objRatio, *iterPrint)
	elapsed := time.Since(start)

	// Report resultant coreset & time taken
	fmt.Printf("%v\n", result)
	fmt.Print("Obtained solution of size ", len(result), " in ")
	fmt.Printf("%s\n", elapsed)
}
