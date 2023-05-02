package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/lib/pq"
	_ "github.com/lib/pq"
)

var (
	dbFlag    = flag.String("db", "", "Name of the PostgreSQL database")
	tableFlag = flag.String("table", "", "Name of the table to store the data")
	fileFlag  = flag.String("file", "", "Path of the txt file")
	rowsFlag  = flag.Int("n", 0, "Number of rows in the database")
	batchFlag = flag.Int("batch", 1000, "Batch size for inserting rows")
)

func main() {
	flag.Parse()

	connStr := fmt.Sprintf("dbname=%s sslmode=disable", *dbFlag)
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	file, err := os.Open(*fileFlag)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Create table if it does not exist
	createTableQuery := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id INTEGER,
			pl INTEGER[]
		);
	`, *tableFlag)

	_, err = db.Exec(createTableQuery)
	if err != nil {
		panic(err)
	}

	// Truncate table to remove existing data
	truncateTableQuery := fmt.Sprintf("TRUNCATE TABLE %s;", *tableFlag)
	_, err = db.Exec(truncateTableQuery)
	if err != nil {
		panic(err)
	}

	tx, err := db.Begin()
	if err != nil {
		panic(err)
	}

	batchCounter := 0
	insertQuery := fmt.Sprintf("INSERT INTO %s (id, pl) VALUES ($1, $2)", *tableFlag)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		parts := strings.Split(line, " : { ")
		idStr := parts[0]
		plStr := strings.TrimRight(parts[1], " }")

		id, err := strconv.Atoi(idStr)
		if err != nil {
			panic(err)
		}

		plStr = strings.ReplaceAll(plStr, " ", "")
		plArray := strings.Split(plStr, ",")
		pl := make([]int, len(plArray))
		for i, v := range plArray {
			pl[i], err = strconv.Atoi(v)
			if err != nil {
				panic(err)
			}
		}

		_, err = tx.Exec(insertQuery, id, pq.Array(pl))
		if err != nil {
			panic(err)
		}

		batchCounter++
		if batchCounter >= *batchFlag {
			err = tx.Commit()
			if err != nil {
				panic(err)
			}

			tx, err = db.Begin()
			if err != nil {
				panic(err)
			}
			batchCounter = 0
		}
	}

	if batchCounter > 0 {
		err = tx.Commit()
		if err != nil {
			panic(err)
		}
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	fmt.Println("Data has been loaded into the PostgreSQL database.")
}
