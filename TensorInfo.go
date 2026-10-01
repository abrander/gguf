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
// If the tensor data is empty, it will return io.ErrUnexpectedEOF.
func (t *TensorInfo) Reader() (io.Reader, error) {
	if readerAt, ok := t.g.r.(io.ReaderAt); ok {
		offset := t.DataOffset()

		const maxInt64 = int64(^uint64(0) >> 1)

		length := maxInt64 - offset
		if length < 0 {
			length = 0
		}

		if length == 0 {
			return nil, io.ErrUnexpectedEOF
		}

		return io.NewSectionReader(readerAt, offset, length), nil
	}

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
// useful in combination with TensorSize() on the Reader if you
// would like to show a progress bar.
// 0 will be returned if the tensor type is unknown or if the tensor
// has no data.
func (t *TensorInfo) Size() int64 {
	s, found := sizes[t.Type]
	if !found {
		return 0
	}

	values := uint64(1)

	for _, d := range t.Dimensions {
		values *= d
	}

	return int64((values / s.valuesinblock) * s.blocksize)
}

// Path returns the name of the file holding the tensor, if it was opened
// by OpenFile. In a split model, that's not necessarily the file given
// to OpenFile.
func (t *TensorInfo) Path() string {
	return t.g.path
}

// DataOffset returns the offset of the tensor data within the GGUF
// file returned by Path().
func (t *TensorInfo) DataOffset() int64 {
	return int64(t.g.tensorOffset) + int64(t.Offset)
}
