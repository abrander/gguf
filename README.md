# package gguf

[![Go Reference](https://pkg.go.dev/badge/github.com/abrander/gguf.svg)](https://pkg.go.dev/github.com/abrander/gguf)

This is a Go package for reading GGUF files.

The package is only concerned with reading the metadata and the tensor bytes. It
does not interpret the data in any way.

GGUF versions 1, 2 and 3 are supported.

## Installation

```bash
go get github.com/abrander/gguf
```

## Usage example

```go
package main

import (
	"fmt"
	"os"

	"github.com/abrander/gguf"
)

func readTensor(t gguf.TensorInfo) {
	r, _ := t.Reader()

	// TODO: Read the actual tensor data...
}

func main() {
	g, _ := gguf.OpenFile("llama-2-7b-chat.Q4_0.gguf")

    fmt.Printf("Got GGUF file version: %d\n", g.Version)

	arch, _ := g.Metadata.String("general.architecture")
	if arch != "llama" {
		os.Exit(1)
	}

	contextLength, _ := g.Metadata.Int("llama.context_length")
	fmt.Printf("Context length: %d\n", contextLength)

	for _, t := range g.Tensors {
		readTensor(t)
	}
}
```

## ggufmeta

The package comes with a command line tool for inspecting GGUF files.

```bash
$ go install github.com/abrander/gguf/ggufmeta@latest
$ ggufmeta llama-2-7b-chat.Q4_0.gguf
```
