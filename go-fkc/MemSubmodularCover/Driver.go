package main

import (
	"flag"
	"fmt"
)

func main() {
	// Define command-line flags
	dbType := flag.String("dbtype", "mongo", "Database software where data is stored")
	dbFlag := flag.String("db", "dummydb", "MongoDB DB")
	collectionFlag := flag.String("col", "n1000d3m5r20", "ollection containing points")
	coverageFlag := flag.Int("k", 20, "k-coverage requirement")
	groupReqFlag := flag.Int("g", 100, "group count requirement")
	groupCntFlag := flag.Int("m", 5, "number of groups")
	optimFlag := flag.Int("optim", 0, "optimization mode")
	threadsFlag := flag.Int("t", 1, "number of threads")
	cardinalityFlag := flag.Int("c", -1, "cardinality constraint")
	dense := flag.Bool("dense", true, "whether the graph is denser than the k-Coverage requirement")
	eps := flag.Float64("eps", 0.1, "parameter used for optimization algorithms")
	//objRatio := flag.Float64("objratio", 0.9, "portion of objective function to be satisfied with LazyLazy before switching to Lazy")
	iterPrint := flag.Bool("iterprint", true, "whether to report each iteration's progress")
	groupFile := flag.String("groupfile", "", "file where group labels are stored (for psql)")

	// Parse all flags
	flag.Parse()

	// Make the groupReqs array
	groupReqs := make([]int, *groupCntFlag)
	for i := 0; i < *groupCntFlag; i++ {
		groupReqs[i] = *groupReqFlag
	}

	// Run submodularCover
	coreset, funcVal := SubmodularCover(*dbType, *dbFlag, *collectionFlag, *coverageFlag, groupReqs, *optimFlag, *threadsFlag, *cardinalityFlag, *dense, *eps, *iterPrint)

	// Report resultant coreset & time taken
	//fmt.Println("elapsed:", elapsed)
	fmt.Println("size:", len(coreset))
	fmt.Println("function value:", funcVal)
	//fmt.Println("coreset:", coreset)
}
