package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// V1: https://github.com/philpax/ggml/blob/2b65fba00c83b9fa041df2ac55ccd8c2f10c5281/docs/gguf.md
// V2: https://github.com/philpax/ggml/blob/574b408f472923071fbc7a265c974c00ce01f959/docs/gguf.md
// V3: Like v2, but can store both little- and big-endian data.

// Reader is a reader for GGUF files.
type Reader struct {
	r io.ReadSeeker

	// ByteOrder is the byte order of the GGUF file. This package
	// does not do any byte swapping for tensor data, it's the
	// responsibility of you, the user, to make sure endian is
	// correct or swapped.
	ByteOrder binary.ByteOrder

	// Version is the GGUF version.
	Version int

	// Metadata is the metadata in the file.
	Metadata Metadata

	// Tensors is the list of tensors in the file, or in all the files
	// of a split model.
	Tensors []TensorInfo

	tensorOffset int64

	// path and file are set when the file was opened by OpenFile.
	path string
	file *os.File

	// parts are the rest of the files of a split model.
	parts []*Reader

	// data is the memory mapping of the file, made by the first call to
	// Bytes for one of its tensors.
	mu     sync.Mutex
	data   []byte
	closed bool

	// Helper to read int32 or int64 depending on GGUF version.
	readUint func(io.Reader, binary.ByteOrder) (uint64, error)
}

// readString reads a GGUF string from r.
func (r *Reader) readString() (string, error) {
	trim := func(r rune) bool {
		return r == 0
	}

	if r.Version == 1 {
		trim = func(r rune) bool {
			var asciiSpace = [33]bool{
				0:    true, // null character
				'\t': true, // horizontal tab
				'\n': true, // new line
				'\v': true, // vertical tab
				'\f': true, // form feed
				'\r': true, // carriage return
				' ':  true, // space
			}

			if int(r) < len(asciiSpace) && asciiSpace[r] {
				return true
			}

			return false
		}
	}

	length, err := r.readUint(r.r, r.ByteOrder)
	if err != nil {
		return "", err
	}

	data := make([]byte, length)

	_, err = io.ReadFull(r.r, data)
	if err != nil {
		return "", err
	}

	datastr := strings.TrimFunc(string(data), trim)

	return string(datastr), nil
}

// readMetaDataValueScalar reads a GGUF scalar value from r. String is a special
// case because it is variable length.
func (r *Reader) readMetaDataValueScalar(typ Type) (interface{}, error) {
	switch typ {
	case Uint8:
		return read[uint8](r.r, r.ByteOrder)

	case Int8:
		return read[int8](r.r, r.ByteOrder)

	case Uint16:
		return read[uint16](r.r, r.ByteOrder)

	case Int16:
		return read[int16](r.r, r.ByteOrder)

	case Uint32:
		return read[uint32](r.r, r.ByteOrder)

	case Int32:
		return read[int32](r.r, r.ByteOrder)

	case Float32:
		return read[float32](r.r, r.ByteOrder)

	case Bool:
		i, err := read[uint8](r.r, r.ByteOrder)

		if i != 0 && i != 1 {
			return nil, fmt.Errorf("invalid bool value: %d", i)
		}

		return i == 1, err

	case String:
		return r.readString()

	case Uint64:
		return read[uint64](r.r, r.ByteOrder)

	case Int64:
		return read[int64](r.r, r.ByteOrder)

	case Float64:
		return read[float64](r.r, r.ByteOrder)

	default:
		return nil, fmt.Errorf("invalid scalar type: %d", typ)
	}
}

// readMetaDataValueArray reads a GGUF metadata array from r.
func readMetaDataValueArray[T readables](r *Reader, length uint64) ([]T, error) {
	a := make([]T, length)

	for i := uint64(0); i < length; i++ {
		v, err := read[T](r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		a[i] = v
	}

	return a, nil
}

// readMetaValue reads a GGUF metadata value from r.
func (r *Reader) readMetaValue() (interface{}, error) {
	typ, err := read[Type](r.r, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	switch typ {
	case Array:
		aType, err := read[Type](r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		length, err := r.readUint(r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		switch aType {
		case Uint8:
			return readMetaDataValueArray[uint8](r, length)

		case Int8:
			return readMetaDataValueArray[int8](r, length)

		case Uint16:
			return readMetaDataValueArray[uint16](r, length)

		case Int16:
			return readMetaDataValueArray[int16](r, length)

		case Uint32:
			return readMetaDataValueArray[uint32](r, length)

		case Int32:
			return readMetaDataValueArray[int32](r, length)

		case Float32:
			return readMetaDataValueArray[float32](r, length)

		case Bool:
			a, err := readMetaDataValueArray[uint8](r, length)
			if err != nil {
				return nil, err
			}

			b := make([]bool, length)

			for i, v := range a {
				if v != 0 && v != 1 {
					return nil, fmt.Errorf("invalid bool value: %d", v)
				}

				b[i] = v == 1
			}

			return b, nil

		case String:
			a := make([]string, length)

			for i := uint64(0); i < length; i++ {
				v, err := r.readString()
				if err != nil {
					return nil, err
				}

				a[i] = v
			}

			return a, nil

		case Uint64:
			return readMetaDataValueArray[uint64](r, length)

		case Int64:
			return readMetaDataValueArray[int64](r, length)

		case Float64:
			return readMetaDataValueArray[float64](r, length)

		default:
			return nil, fmt.Errorf("unsupported array type: %d", aType)
		}

	default:
		return r.readMetaDataValueScalar(Type(typ))
	}
}

// OpenFile opens a GGUF file. If it's the first file of a model split in
// several files, like model-00001-of-00004.gguf, the rest are opened too,
// and Tensors holds the tensors of all of them. Close closes all the
// files.
func OpenFile(filename string) (*Reader, error) {
	r, err := openFile(filename)
	if err != nil {
		return nil, err
	}

	err = r.openSplit()
	if err != nil {
		r.Close()

		return nil, err
	}

	return r, nil
}

// openFile opens a single GGUF file.
func openFile(filename string) (*Reader, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	r, err := Open(f)
	if err != nil {
		f.Close()

		return nil, fmt.Errorf("%s: %w", filename, err)
	}

	r.path = filename
	r.file = f

	return r, nil
}

// Bytes returns the data of tensor t. The file holding it is memory mapped
// the first time Bytes is called for one of its tensors, so nothing is
// copied. r must be opened by OpenFile, or by Open with an *os.File.
func (r *Reader) Bytes(t *TensorInfo) (Data, error) {
	part := t.g
	if part != r && !r.hasPart(part) {
		return nil, fmt.Errorf("tensor %q is not from this reader", t.Name)
	}

	data, err := part.mapped()
	if err != nil {
		return nil, err
	}

	values := int64(1)
	for _, d := range t.Dimensions {
		values *= int64(d)
	}

	size, err := t.Type.ByteSize(values)
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
	}

	offset := t.DataOffset()

	if offset+size > int64(len(data)) {
		return nil, fmt.Errorf("tensor %q ends after the end of the file", t.Name)
	}

	return Data(data[offset : offset+size : offset+size]), nil
}

func (r *Reader) hasPart(part *Reader) bool {
	for _, p := range r.parts {
		if p == part {
			return true
		}
	}

	return false
}

// mapped returns the mapping of the file, mapping it if needed.
func (r *Reader) mapped() ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, errors.New("reader is closed")
	}

	if r.data != nil {
		return r.data, nil
	}

	f, ok := r.r.(*os.File)
	if !ok {
		return nil, errors.New("only files can be mapped")
	}

	data, err := mapFile(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", f.Name(), err)
	}

	r.data = data

	return data, nil
}

// Close unmaps the files mapped by Bytes, and closes the files opened by
// OpenFile. Readers created by Open are left alone, closing them is up to
// the caller.
func (r *Reader) Close() error {
	var errs []error

	for _, part := range append([]*Reader{r}, r.parts...) {
		part.mu.Lock()

		errs = append(errs, unmapFile(part.data))
		part.data = nil
		part.closed = true

		part.mu.Unlock()

		if part.file != nil {
			errs = append(errs, part.file.Close())
			part.file = nil
		}
	}

	return errors.Join(errs...)
}

// Open opens a GGUF file from r. r must be positoned at the start
// of the file.
// This will not open the rest of the files if it's part of a split
// model. Use OpenFile instead.
func Open(readseeker io.ReadSeeker) (*Reader, error) {
	var buf [4]byte

	_, err := readseeker.Read(buf[:])
	if err != nil {
		return nil, err
	}

	if !bytes.Equal(buf[:], []byte(magic)) {
		return nil, fmt.Errorf("not a GGUF file, unknown magic: %q", buf)
	}

	// Jump to the last byte of the version, and check if this
	// could be a big-endian file.
	_, err = readseeker.Seek(3, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	bigEndianMarker := int8(0)
	err = binary.Read(readseeker, binary.LittleEndian, &bigEndianMarker)
	if err != nil {
		return nil, err
	}

	var byteOrder binary.ByteOrder = binary.LittleEndian

	if bigEndianMarker != 0 {
		byteOrder = binary.BigEndian
	}

	// Jump back to read the version in file byteorder.
	_, err = readseeker.Seek(-4, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	version, err := read[uint32](readseeker, byteOrder)
	if err != nil {
		return nil, err
	}

	r := &Reader{
		r:         readseeker,
		ByteOrder: byteOrder,
		Version:   int(version),
	}

	switch version {
	case 1:
		r.readUint = readCast[uint32, uint64]

	case 2, 3:
		r.readUint = read[uint64]

	default:
		return nil, fmt.Errorf("invalid version: %d", version)
	}

	tensorCount, err := r.readUint(readseeker, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	metadataCount, err := r.readUint(readseeker, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	r.Metadata = make(map[string]interface{})

	for i := uint64(0); i < metadataCount; i++ {
		name, err := r.readString()
		if err != nil {
			return nil, err
		}

		value, err := r.readMetaValue()
		if err != nil {
			return nil, err
		}

		if u, ok := value.(uint32); ok && name == "general.file_type" {
			value = Filetype(u)
		}

		r.Metadata[name] = value
	}

	alignment := defaultAlignment

	if a, found := r.Metadata["general.alignment"]; found {
		switch v := a.(type) {
		case uint32:
			alignment = int64(v)

		default:
			return nil, fmt.Errorf("invalid alignment type: %T", a)
		}
	}

	r.Tensors = make([]TensorInfo, tensorCount)

	for i := uint64(0); i < tensorCount; i++ {
		r.Tensors[i].g = r

		r.Tensors[i].Name, err = r.readString()
		if err != nil {
			return nil, err
		}

		nDimensions, err := read[uint32](readseeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		r.Tensors[i].Dimensions = make([]uint64, nDimensions)

		for j := uint32(0); j < nDimensions; j++ {
			r.Tensors[i].Dimensions[j], err = r.readUint(readseeker, r.ByteOrder)
			if err != nil {
				return nil, err
			}
		}

		typ, err := read[uint32](readseeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		r.Tensors[i].Type = GGML(typ)

		r.Tensors[i].Offset, err = r.readUint(readseeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}
	}

	current, err := readseeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	r.tensorOffset = (current + alignment - 1) / alignment * alignment

	return r, nil
}

// TensorInfo returns the tensor info for the tensor with the given
// name. If the tensor is not found, an error is returned.
func (r *Reader) TensorInfo(name string) (*TensorInfo, error) {
	for i := range r.Tensors {
		if r.Tensors[i].Name == name {
			return &r.Tensors[i], nil
		}
	}

	return nil, fmt.Errorf("tensor %q not found", name)
}

// TensorSize returns the total size of all tensors in the file.
// This is useful if you would like to show a progress bar showing
// the progress of reading the file.
func (r *Reader) TensorSize() int64 {
	size := int64(0)

	for _, t := range r.Tensors {
		size += t.Size()
	}

	return size
}
