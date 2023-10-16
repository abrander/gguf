package gguf

import (
	"io"
)

type TensorInfo struct {
	g *Reader

	// The name of the tensor.
	Name string

	Dimensions []uint64

	Type GGML

	Offset uint64
}

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
