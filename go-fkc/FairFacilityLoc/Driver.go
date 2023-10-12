package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func main() {
	// Parse command-line flag
	configFlag := flag.String("config", "config.csv", "Config CSV file")
	flag.Parse()

	// Parse the flag CSV file
	reader := readCSVFile(*configFlag)
	configs := parseCSVFile(reader)

	// Loop for each experiment defined in the configs
	for i := 0; i < len(configs); i++ {
		config := configs[i]
		// Grab all the arguments
		var (
			ok           bool
			db           string
			collection   string
			groupReq     int
			groupCnt     int
			optim        string
			threads      int
			cardinality  int
			iterPrint    bool
			resultDest   string
			partialGraph bool
			slices       []int
			ssSize       int
		)

		if db, ok = config["DB"].(string); !ok {
			fmt.Printf("DB parse error: %v (%T)\n", config["DB"], config["DB"])
			db = "WrongDB"
		}
		if collection, ok = config["Collection"].(string); !ok {
			fmt.Printf("table parse error: %v (%T)\n", config["Collection"], config["Collection"])
			collection = "WrongTable"
		}
		if groupReqStr, ok := config["GroupReq"].(string); !ok {
			fmt.Println("GroupReq parse error")
			groupReq = -1
		} else {
			groupReq, _ = strconv.Atoi(groupReqStr)
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
		if iterPrintStr, ok := config["IterPrint"].(string); !ok {
			fmt.Println("IterPrint parse error")
			iterPrint = true
		} else {
			iterPrint, _ = strconv.ParseBool(iterPrintStr)
		}
		if resultDest, ok = config["ResultDest"].(string); !ok {
			fmt.Println("ResultDest parse error")
			resultDest = "./"
		}
		if partialGraphStr, ok := config["partialGraph"].(string); !ok {
			fmt.Println("partial graph parse error")
			partialGraph = false
		} else {
			partialGraph, _ = strconv.ParseBool(partialGraphStr)
		}

		if ssSizeStr, ok := config["ssSize"].(string); !ok {
			fmt.Println("Subset size parse error")
			ssSize = 0
		} else {
			ssSize, _ = strconv.Atoi(ssSizeStr)
		}
		if slicesStr, ok := config["slices"].(string); !ok {
			fmt.Println("Slices parse error")
			slices = make([]int, 0)
		} else {
			slices = parseSlices(slicesStr, ssSize)
		}

		// Report the configs to stdout
		fmt.Println(db, collection, groupReq, groupCnt, optim, threads,
			cardinality, iterPrint, resultDest, partialGraph, ssSize, slices)

		coreset, funcVal, preTime, inTime := SubmodularCover(db, collection,
			groupReq, groupCnt, optim, threads, cardinality, iterPrint, partialGraph, slices, ssSize)

		// Run SubmodularCover

		// Turn result into string
		result := "PreprocessingTimeHuman: " + preTime.String() + "\n"
		result += "InprocessingTimeHuman: " + inTime.String() + "\n"
		result += "PreprocessingTimeSecs: " + strconv.Itoa(int(preTime.Seconds())) + "\n"
		result += "InprocessingTimeSecs: " + strconv.Itoa(int(inTime.Seconds())) + "\n"
		result += "CoresetSize: " + strconv.Itoa(len(coreset)) + "\n"
		result += "FunctionValue: " + strconv.FormatFloat(funcVal, 'f', 6, 64) + "\n"

		// Report result to stdout and/or file
		fmt.Println(result)
		result += "Coreset:\n"
		for j := 0; j < len(coreset); j++ {
			result += strconv.Itoa(coreset[j]) + "\n"
		}
		if resultDest != "stdout" {
			fileName := db + "_" + collection + "_f" + strconv.Itoa(groupReq) + "_" + optim + "_k" + "_t" + strconv.Itoa(threads) + "_c" + strconv.Itoa(cardinality)
			writeToFile(result, resultDest+"/"+fileName)
		}
	}
}

func parseSlices(slices string, ssSize int) []int {

	slicesArr := strings.Split(slices, " ")

	slicesIntArr := make([]int, ssSize)

	for i := 0; i < ssSize; i++ {

		slicesIntArr[i], _ = strconv.Atoi(slicesArr[i])

	}

	return slicesIntArr

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
