package gguf

import (
	"encoding/binary"
	"io"
)

// read reads a value of type T from a binary stream.
func read[T Number](r io.Reader, byteorder binary.ByteOrder) (T, error) {
	var v T

	err := binary.Read(r, byteorder, &v)

	return v, err
}

// readCast reads a value of type T from a binary stream and casts it to type C.
func readCast[T, C Number](r io.Reader, byteorder binary.ByteOrder) (C, error) {
	var v T

	err := binary.Read(r, byteorder, &v)

	return C(v), err
}
