package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)

func main() {
	// Parse command-line flag
	configFlag := flag.String("config", "config.csv", "CSV where flags are stored")
	flag.Parse()

	// Parse the flag CSV file
	reader := readCSVFile(*configFlag)
	configs := parseCSVFile(reader)

	// Loop for each experiment defined in the configs
	for i := 0; i < len(configs); i++ {
		config := configs[i]
		// Grab all the arguments
		var (
			ok          bool
			dbType      string
			db          string
			table       string
			coverage    int
			groupCnt    int
			optim       string
			threads     int
			cardinality int
			dense       bool
			eps         float64
			iterPrint   bool
			groupFile   string
			resultDest  string
		)
		if dbType, ok = config["DBType"].(string); !ok {
			fmt.Printf("dbType parse error: %v (%T)\n", config["DBType"], config["DBType"])
			dbType = "WrongDBType"
		}
		if db, ok = config["DB"].(string); !ok {
			fmt.Printf("DB parse error: %v (%T)\n", config["DB"], config["DB"])
			db = "WrongDB"
		}
		if table, ok = config["Table"].(string); !ok {
			fmt.Printf("table parse error: %v (%T)\n", config["Table"], config["Table"])
			table = "WrongTable"
		}
		if coverageStr, ok := config["Coverage"].(string); !ok {
			fmt.Println("coverage parse error")
			coverage = -1
		} else {
			coverage, _ = strconv.Atoi(coverageStr)
		}
		if groupCntStr, ok := config["GroupCnt"].(string); !ok {
			fmt.Println("GroupCnt parse error")
			groupCnt = -1
		} else {
			groupCnt, _ = strconv.Atoi(groupCntStr)
		}
		if optim, ok = config["Optim"].(string); !ok {
			fmt.Println("Optim parse error")
			optim = "WrongOptim"
		} 
		if threadsStr, ok := config["Threads"].(string); !ok {
			fmt.Println("Threads parse error")
			threads = -1
		} else {
			threads, _ = strconv.Atoi(threadsStr)
		}
		if cardinalityStr, ok := config["Cardinality"].(string); !ok {
			fmt.Println("Cardinality parse error")
			cardinality = -1
		} else {
			cardinality, _ = strconv.Atoi(cardinalityStr)
		}
		if denseStr, ok := config["Dense"].(string); !ok {
			fmt.Println("Dense parse error")
			dense = false
		} else {
			dense, _ = strconv.ParseBool(denseStr)
		}
		if epsStr, ok := config["Eps"].(string); !ok {
			fmt.Println("Eps parse error")
			eps = 0.1
		} else {
			eps, _ = strconv.ParseFloat(epsStr, 64)
		}
		if iterPrintStr, ok := config["IterPrint"].(string); !ok {
			fmt.Println("IterPrint parse error")
			iterPrint = true
		} else {
			iterPrint, _ = strconv.ParseBool(iterPrintStr)
		}
		if groupFile, ok = config["GroupFile"].(string); !ok {
			fmt.Println("GroupFile parse error")
			groupFile = ""
		}
		if resultDest, ok = config["ResultDest"].(string); !ok {
			fmt.Println("ResultDest parse error")
			resultDest = "./"
		}
		// Parse the GroupReq argument
		groupReqs, groupReqStr := parseGroupReqs(config, groupCnt, groupFile, dbType, db, table)

		fmt.Println(dbType, db, table, coverage, groupCnt, optim, threads, cardinality, dense, eps, iterPrint, groupFile, resultDest, groupReqs, groupReqStr)

		// Run SubmodularCover
		coreset, funcVal, preTime, inTime := SubmodularCover(dbType, db, table, coverage, groupReqs, optim, threads, cardinality, dense, eps, iterPrint, groupFile)

		// Turn result into string
		result := "PreprocessingTimeHuman: " + preTime.String() + "\n"
		result += "InprocessingTimeHuman: " + inTime.String() + "\n"
		result += "PreprocessingTimeSecs: " + strconv.Itoa(int(preTime.Seconds())) + "\n"
		result += "InprocessingTimeSecs: " + strconv.Itoa(int(inTime.Seconds())) + "\n"
		result += "CoresetSize: " + strconv.Itoa(len(coreset)) + "\n"
		result += "FunctionValue: " + strconv.Itoa(funcVal) + "\n"
		if !iterPrint {
			result += "Coreset: "
			for j := 0; j < len(coreset); j++ {
				result += strconv.Itoa(coreset[j]) + ", "
			}
		}

		// Report result to stdout and/or file
		fmt.Println(result)
		if resultDest != "stdout" {
			fileName := db + "_" + table + "_" + optim + "_k" + strconv.Itoa(coverage) + "_f" + groupReqStr + "_t" + strconv.Itoa(threads) + "_c" + strconv.Itoa(cardinality) + "_eps" + strconv.FormatFloat(eps, 'f', -1, 64)
			writeToFile(result, resultDest+"/"+fileName)
		}
	}
}

func readCSVFile(filename string) *csv.Reader {
	file, err := os.Open(filename)
	handleError(err)

	reader := csv.NewReader(file)
	return reader
}

func parseCSVFile(reader *csv.Reader) []map[string]interface{} {
	result := make([]map[string]interface{}, 0)

	header, err := reader.Read()
	handleError(err)
	for i := 0; ; i++ {
		record, err := reader.Read()
		// Break if EOF
		if err == io.EOF {
			break
		}
		handleError(err)

		// Parse each line
		flags := make(map[string]interface{}, 15)
		for j, value := range record {
			flags[header[j]] = value
		}
		result = append(result, flags)
	}
	return result
}

func parseGroupReqs(config map[string]interface{}, groupCnt int, groupFile string, dbType string, db string, table string) ([]int, string) {
	groupReqStr := config["GroupReq"].(string)
	groupReqs := make([]int, groupCnt)
	if groupReq, err := strconv.Atoi(groupReqStr); err == nil { // Integer type
		for j := 0; j < groupCnt; j++ {
			groupReqs[j] = groupReq
		}
	} else {
		totalPercentage, _ := strconv.Atoi(groupReqStr[:len(groupReqStr)-1])
		groupReqs = proportionalGroupReqs(dbType, db, table, totalPercentage, groupCnt, groupFile)
	}
	return groupReqs, groupReqStr
}

func writeToFile(content string, fileName string) {
	// Get file
	file, err := os.Create(fileName)
	handleError(err)
	defer file.Close()
	// Get writer & write string
	writer := bufio.NewWriter(file)
	_, err = writer.WriteString(content)
	handleError(err)
	// Flush writer
	writer.Flush()
}

func proportionalGroupReqs(dbType string, db string, table string, totalPercentage int, groupCnt int, groupFile string) []int {
	switch dbType {
	case "mongo":
		return getMongoProportionalGroupReqs(db, table, totalPercentage, groupCnt)
	case "psql":
		return getPostgresProportionalGroupReqs(groupFile, totalPercentage, groupCnt)
	default:
		return []int{}
	}
}

func getMongoProportionalGroupReqs(dbName string, table string, totalPercentage int, groupCnt int) []int {
	// Get collection
	collection := getMongoCollection(dbName, table)
	n := getCollectionSize(collection)
	gt := make([]int, groupCnt)
	// Get cursor
	cur := getFullCursor(collection)
	defer cur.Close(context.Background())
	// Iterate over cursor
	for cur.Next(context.Background()) {
		point := getEntryFromCursor(cur)
		group := point.Group
		gt[group] += 1
	}
	// Compute GroupReqs
	groupReqs := make([]int, groupCnt)
	for g := 0; g < groupCnt; g++ {
		proportion := float64(gt[g]) / float64(n)
		groupReq := int(math.Round(float64(n) * 0.01 * float64(totalPercentage) * proportion))
		groupReqs[g] = groupReq
	}
	return groupReqs
}

func getPostgresProportionalGroupReqs(groupFileName string, totalPercentage int, groupCnt int) []int {
	// Get group txt file
	groupFileScanner, groupFileFile := getFileScanner(groupFileName)
	defer groupFileFile.Close()
	gt := make([]int, groupCnt)
	n := 0
	for ; ; n++ {
		// Check if EOF
		_, err := groupFileScanner.Peek(1)
		if err != nil {
			break
		}
		// Increment gt count
		gr := parseGroupLine(groupFileScanner, n)
		gt[gr] += 1
	}
	// Compute GroupReqs
	groupReqs := make([]int, groupCnt)
	for g := 0; g < groupCnt; g++ {
		proportion := float64(gt[g]) / float64(n)
		groupReq := int(math.Round(float64(n) * 0.01 * float64(totalPercentage) * proportion))
		groupReqs[g] = groupReq
	}
	return groupReqs
}
