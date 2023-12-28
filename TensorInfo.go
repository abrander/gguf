package gguf

import (
	"io"
)

// TensorInfo is used to represent a tensor in a GGUF file.
type TensorInfo struct {
	g *Reader

	// The name of the tensor.
	Name string

	Dimensions []uint64

	Type GGML

	Offset uint64
}

// Reader returns an io.Reader that can be used to read the tensor
// data. The reader is positioned at the start of the tensor data.
// The caller of this function is responsible for calculating how
// much data to read.
func (t *TensorInfo) Reader() (io.Reader, error) {
	// FIXME: Use io.NewSectionReader.
	_, err := t.g.r.Seek(t.g.tensorOffset, io.SeekStart)
	if err != nil {
		return nil, err
	}

	_, err = t.g.r.Seek(int64(t.Offset), io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	return t.g.r, nil
}

// Size returns the size of the tensor data in bytes. This can be
// useful in comfination with TensorSize() on the Reader if you
// would like to show a progress bar.
func (t *TensorInfo) Size() int64 {
	s, found := sizes[t.Type]
	if !found {
		panic("unknown type: " + t.Type.String())
	}

	values := uint64(1)

	for _, d := range t.Dimensions {
		values *= d
	}

	return int64((values / s.valuesinblock) * s.blocksize)
}
