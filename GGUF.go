package gguf

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
)

// V1: https://github.com/philpax/ggml/blob/2b65fba00c83b9fa041df2ac55ccd8c2f10c5281/docs/gguf.md
// V2: https://github.com/philpax/ggml/blob/574b408f472923071fbc7a265c974c00ce01f959/docs/gguf.md

// GGUF is a struct that represents a GGUF file.
type GGUF struct {
	r io.ReadSeeker

	// Version is the GGUF version.
	Version int

	// Metadata is the metadata in the file.
	Metadata map[string]interface{}

	// Tensors is the list of tensors in the file.
	Tensors []TensorInfo

	tensorOffset int64

	// Helper to read int32 or int64 depending on GGUF version.
	readUint func(io.Reader) (uint64, error)
}

// magic is the ftype for GGUF files.
const magic = "GGUF"

// readString reads a GGUF string from r.
func (g *GGUF) readString(r io.Reader) (string, error) {
	trim := func(r rune) bool {
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

	length, err := g.readUint(r)
	if err != nil {
		return "", err
	}

	data := make([]byte, length)

	_, err = io.ReadFull(r, data)
	if err != nil {
		return "", err
	}

	datastr := strings.TrimFunc(string(data), trim)

	return string(datastr), nil
}

// readMetaDataValueScalar reads a GGUF scalar value from r. String is a special
// case because it is variable length.
func (g *GGUF) readMetaDataValueScalar(typ Type, r io.Reader) (interface{}, error) {
	switch typ {
	case Uint8:
		return read[uint8](r)

	case Int8:
		return read[int8](r)

	case Uint16:
		return read[uint16](r)

	case Int16:
		return read[int16](r)

	case Uint32:
		return read[uint32](r)

	case Int32:
		return read[int32](r)

	case Float32:
		return read[float32](r)

	case Bool:
		i, err := read[uint8](r)

		if i != 0 && i != 1 {
			return nil, fmt.Errorf("invalid bool value: %d", i)
		}

		return i == 1, err

	case String:
		return g.readString(r)

	case Uint64:
		return read[uint64](r)

	case Int64:
		return read[int64](r)

	case Float64:
		return read[float64](r)

	default:
		return nil, fmt.Errorf("invalid scalar type: %d", typ)
	}
}

// readMetaDataValueArray reads a GGUF metadata array from r.
func readMetaDataValueArray[T readables](g *GGUF, r io.Reader, length uint64) ([]T, error) {
	a := make([]T, length)

	for i := uint64(0); i < length; i++ {
		v, err := read[T](r)
		if err != nil {
			return nil, err
		}

		a[i] = v
	}

	return a, nil
}

// readMetaValue reads a GGUF metadata value from r.
func (g *GGUF) readMetaValue(r io.Reader) (interface{}, error) {
	typ, err := read[Type](r)
	if err != nil {
		return nil, err
	}

	switch typ {
	case Array:
		aType, err := read[Type](r)
		if err != nil {
			return nil, err
		}

		length, err := g.readUint(r)
		if err != nil {
			return nil, err
		}

		switch aType {
		case Uint8:
			return readMetaDataValueArray[uint8](g, r, length)

		case Int8:
			return readMetaDataValueArray[int8](g, r, length)

		case Uint16:
			return readMetaDataValueArray[uint16](g, r, length)

		case Int16:
			return readMetaDataValueArray[int16](g, r, length)

		case Uint32:
			return readMetaDataValueArray[uint32](g, r, length)

		case Int32:
			return readMetaDataValueArray[int32](g, r, length)

		case Float32:
			return readMetaDataValueArray[float32](g, r, length)

		case Bool:
			a, err := readMetaDataValueArray[uint8](g, r, length)
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
				v, err := g.readString(r)
				if err != nil {
					return nil, err
				}

				a[i] = v
			}

			return a, nil

		case Uint64:
			return readMetaDataValueArray[uint64](g, r, length)

		case Int64:
			return readMetaDataValueArray[int64](g, r, length)

		case Float64:
			return readMetaDataValueArray[float64](g, r, length)

		default:
			return nil, fmt.Errorf("unsupported array type: %d", aType)
		}

	default:
		return g.readMetaDataValueScalar(Type(typ), r)
	}
}

// OpenFile opens a GGUF file.
func OpenFile(filename string) (*GGUF, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	return Open(f)
}

// Open opens a GGUF file from r. r must be positoned at the start
// of the file.
func Open(r io.ReadSeeker) (*GGUF, error) {
	var buf [4]byte

	_, err := r.Read(buf[:])
	if err != nil {
		return nil, err
	}

	if !bytes.Equal(buf[:], []byte(magic)) {
		return nil, fmt.Errorf("not a GGUF file, unknown magic: %q", buf)
	}

	version, err := read[uint32](r)
	if err != nil {
		return nil, err
	}

	g := &GGUF{
		r:       r,
		Version: int(version),
	}

	switch version {
	case 1:
		g.readUint = readCast[uint32, uint64]

	case 2:
		g.readUint = read[uint64]

	default:
		return nil, fmt.Errorf("invalid version: %d", version)
	}

	tensorCount, err := g.readUint(r)
	if err != nil {
		return nil, err
	}

	metadataCount, err := g.readUint(r)
	if err != nil {
		return nil, err
	}

	g.Metadata = make(map[string]interface{})

	for i := uint64(0); i < metadataCount; i++ {
		name, err := g.readString(r)
		if err != nil {
			return nil, err
		}

		value, err := g.readMetaValue(r)
		if err != nil {
			return nil, err
		}

		if u, ok := value.(uint32); ok && name == "general.file_type" {
			value = Filetype(u)
		}

		g.Metadata[name] = value
	}

	alignment := int64(32)

	if a, found := g.Metadata["general.alignment"]; found {
		switch v := a.(type) {
		case uint32:
			alignment = int64(v)

		default:
			return nil, fmt.Errorf("invalid alignment type: %T", a)
		}
	}

	g.Tensors = make([]TensorInfo, tensorCount)

	for i := uint64(0); i < tensorCount; i++ {
		g.Tensors[i].g = g

		g.Tensors[i].Name, err = g.readString(r)
		if err != nil {
			return nil, err
		}

		nDimensions, err := read[uint32](r)
		if err != nil {
			return nil, err
		}

		g.Tensors[i].Dimensions = make([]uint64, nDimensions)

		for j := uint32(0); j < nDimensions; j++ {
			g.Tensors[i].Dimensions[j], err = g.readUint(r)
			if err != nil {
				return nil, err
			}
		}

		typ, err := read[uint32](r)
		if err != nil {
			return nil, err
		}

		g.Tensors[i].Type = GGML(typ)

		g.Tensors[i].Offset, err = g.readUint(r)
		if err != nil {
			return nil, err
		}
	}

	current, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	g.tensorOffset = current + alignment - (current % alignment)

	return g, nil
}

func (g *GGUF) TensorInfo(name string) (*TensorInfo, error) {
	for i := range g.Tensors {
		if g.Tensors[i].Name == name {
			return &g.Tensors[i], nil
		}
	}

	return nil, fmt.Errorf("tensor %q not found", name)
}

func MetaValue[T any](g *GGUF, name string) (T, error) {
	var zero T
	v, found := g.Metadata[name]
	if !found {
		return zero, fmt.Errorf("metadata value %q not found", name)
	}

	if _, ok := v.(T); !ok {
		return zero, fmt.Errorf("metadata value %q is not of type %T, type is %T", name, zero, v)
	}

	return v.(T), nil
}

func MetaValueNumber[T ~int | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64](g *GGUF, name string) (T, error) {
	v, found := g.Metadata[name]
	if !found {
		return 0, fmt.Errorf("metadata value %q not found", name)
	}

	switch vv := v.(type) {
	case int:
		return T(vv), nil

	case uint8:
		return T(vv), nil

	case int8:
		return T(vv), nil

	case uint16:
		return T(vv), nil

	case int16:
		return T(vv), nil

	case uint32:
		return T(vv), nil

	case int32:
		return T(vv), nil

	case uint64:
		return T(vv), nil

	case int64:
		return T(vv), nil

	case float32:
		return T(vv), nil

	case float64:
		return T(vv), nil

	default:
		return 0, fmt.Errorf("metadata value %q is not a number, type is %T", name, v)
	}
}
