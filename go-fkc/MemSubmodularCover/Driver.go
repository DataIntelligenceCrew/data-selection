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
		dbType, _ := config["DBType"].(string)
		db, _ := config["DB"].(string)
		table, _ := config["Table"].(string)
		coverage, _ := config["Coverage"].(int)
		groupCnt, _ := config["GroupCnt"].(int)
		optim, _ := config["Optim"].(string)
		threads, _ := config["Threads"].(int)
		cardinality, _ := config["Cardinality"].(int)
		dense, _ := config["Dense"].(bool)
		eps, _ := config["Eps"].(float64)
		iterPrint, _ := config["IterPrint"].(bool)
		groupFile, _ := config["GroupFile"].(string)
		resultDest, _ := config["ResultDest"].(string)
		// Parse the GroupReq argument
		groupReqs, groupReqStr := parseGroupReqs(config["GroupReq"], groupCnt, groupFile, dbType, db, table)

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

func parseGroupReqs(groupReq interface{}, groupCnt int, groupFile string, dbType string, db string, table string) ([]int, string) {
	var groupReqStr string
	groupReqs := make([]int, groupCnt)
	switch groupReq.(type) {
	case int:
		for j := 0; j < groupCnt; j++ {
			groupReqs[j] = groupReq.(int)
		}
		groupReqStr = strconv.Itoa(groupReq.(int))
	case string:
		totalPercentage, err := strconv.Atoi(groupReq.(string)[:len(groupReq.(string))-1])
		handleError(err)
		groupReqs = proportionalGroupReqs(dbType, db, table, totalPercentage, groupCnt, groupFile)
		groupReqStr = groupReq.(string)
	default:
		for j := 0; j < groupCnt; j++ {
			groupReqs[j] = 0
		}
		groupReqStr = "0"
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
