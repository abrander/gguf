package gguf

type TensorInfo struct {
	// The name of the tensor.
	Name string

	Dimensions []uint64

	Type GGML

	Offset uint64
}
