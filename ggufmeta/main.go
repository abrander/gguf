package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/abrander/gguf"
)

func print[T any](name string, val T) {
	fmt.Printf("Metadata: %s: \033[33m%v\033[0m\n", name, val)
}

func printArray[T any](name string, val []T) {
	fmt.Printf("Metadata: %s: [\033[32m%d\033[0m]\033[36m%T\033[0m\n", name, len(val), val[0])
}

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s <file>\n", os.Args[0])
		os.Exit(1)
	}

	g, err := gguf.OpenFile(os.Args[1])
	if err != nil {
		panic(err)
	}

	keys := make([]string, 0, len(g.Metadata))
	for k := range g.Metadata {
		keys = append(keys, k)
	}

	sort.StringSlice(keys).Sort()

	for _, k := range keys {
		v := g.Metadata[k]

		switch vv := v.(type) {
		// Scalars:
		case uint8, int8, uint16, int16, uint32, int32, float32, bool, string, uint64, int64, float64, fmt.Stringer:
			print(k, vv)

		// Arrays:
		case []uint8:
			printArray(k, vv)

		case []int8:
			printArray(k, vv)

		case []uint16:
			printArray(k, vv)

		case []int16:
			printArray(k, vv)

		case []uint32:
			printArray(k, vv)

		case []int32:
			printArray(k, vv)

		case []float32:
			printArray(k, vv)

		case []bool:
			printArray(k, vv)

		case []string:
			printArray(k, vv)

		case []uint64:
			printArray(k, vv)

		case []int64:
			printArray(k, vv)

		case []float64:
			printArray(k, vv)

		default:
			fmt.Printf("%s: %T\n", k, v)
		}
	}

	for _, t := range g.Tensors {
		dims := make([]string, len(t.Dimensions))

		for i, d := range t.Dimensions {
			dims[i] = fmt.Sprintf("\033[32m%d\033[0m", d)
		}

		fmt.Printf("Tensor: %s: \033[36m%s\033[0m [%s]\n", t.Name, t.Type, strings.Join(dims, "×"))
	}
}
