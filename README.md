# package gguf

[![Go Reference](https://pkg.go.dev/badge/github.com/abrander/gguf.svg)](https://pkg.go.dev/github.com/abrander/gguf)

This is a Go package for reading and writing GGUF files.

The package is only concerned with the metadata and the tensor bytes. It does
not interpret the data in any way.

GGUF versions 1, 2 and 3 can be read. Files are written as version 3.

Go 1.27 or later is required.

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

## Writing

`NewWriter` writes the header, with all metadata and tensor infos, to an
`io.WriterAt` like an `*os.File`. That fixes where every tensor goes, so the
tensors can then be written in any order, and from several goroutines at once.
Tensors are found by name, and the type and dimensions must match the header.
Unknown names, mismatching tensors, wrong sizes and tensors written twice are
errors.

```go
tensors := []gguf.TensorInfo{
	{Name: "token_embd.weight", Type: gguf.GgmlF32, Dimensions: []uint64{4096, 32000}},
	{Name: "output.weight", Type: gguf.GgmlQ4_0, Dimensions: []uint64{4096, 32000}},
}

w, _ := gguf.NewWriter(f, gguf.Header{
	Metadata: gguf.Metadata{"general.architecture": "llama"},
	Tensors:  tensors,
})

w.WriteTensor(&tensors[1], quantized) // []byte of exactly the right size
w.CopyTensor(&tensors[0], embeddings) // or from an io.Reader
w.Close()                             // fails if a tensor is missing
```

`w.TensorInfo(name)` looks up a tensor by name, like `Reader.TensorInfo`.

Until `Close` succeeds the file looks valid, with zeros for missing data, so
write to a temporary file and rename it afterwards.

Metadata values must be of the Go types the reader returns (`uint32`,
`[]string` and so on). `Metadata.Set` and `Metadata.SetArray` make sure of
that at compile time, the GGUF type follows from the Go type:

```go
meta.Set("general.architecture", "llama")
meta.Set[uint32]("llama.context_length", 4096)
meta.SetArray[int32]("tokenizer.ggml.token_type", []int32{1, 1, 3})
```

Types derived from them, like `type layers uint32`, are stored as the type
they're derived from. A Go `int` has no GGUF type, and doesn't compile.

`Metadata.Value` returns a value together with its type, and
`Metadata.SetValue` sets one, which is handy for merging metadata from
several files. `gguf.TensorCopy` copies a tensor from a
`Reader` to a `Writer`:

```go
for i := range r.Tensors {
	gguf.TensorCopy(w, &r.Tensors[i])
}
```

`gguf.Copy` rewrites a file, or a split model as a single file, with new
metadata:

```go
r, _ := gguf.OpenFile("model.gguf")
r.Metadata["general.name"] = "renamed"
gguf.Copy(out, r, r.Metadata)
```

## ggufmeta

The package comes with a command line tool for inspecting GGUF files.

```bash
$ go install github.com/abrander/gguf/ggufmeta@latest
$ ggufmeta llama-2-7b-chat.Q4_0.gguf
```
