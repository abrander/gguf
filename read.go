package gguf

import (
	"encoding/binary"
	"io"
)

// readables is a type that can be read from a binary stream by read() and readCast().
type readables interface {
	~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64
}

// read reads a value of type T from a binary stream.
func read[T readables](r io.Reader) (T, error) {
	var v T

	err := binary.Read(r, binary.LittleEndian, &v)

	return v, err
}

// readCast reads a value of type T from a binary stream and casts it to type C.
func readCast[T, C readables](r io.Reader) (C, error) {
	var v T

	err := binary.Read(r, binary.LittleEndian, &v)

	return C(v), err
}
