package gguf

import (
	"encoding/binary"
	"io"
)

// encoder writes GGUF values to w, remembering the first error.
type encoder struct {
	w     io.Writer
	order binary.ByteOrder
	err   error
}

// write writes a fixed-size value, or a slice of them.
func (e *encoder) write(v interface{}) {
	if e.err == nil {
		e.err = binary.Write(e.w, e.order, v)
	}
}

func (e *encoder) writeString(s string) {
	e.write(uint64(len(s)))

	if e.err == nil {
		_, e.err = io.WriteString(e.w, s)
	}
}

// writeValue writes a metadata value, prefixed by its type.
func (e *encoder) writeValue(v interface{}) {
	typ, elem, err := typeOf(v)
	if err != nil {
		if e.err == nil {
			e.err = err
		}

		return
	}

	e.write(typ)

	switch typ {
	case String:
		e.writeString(v.(string))

	case Array:
		e.write(elem)

		switch a := v.(type) {
		case []string:
			e.write(uint64(len(a)))

			for _, s := range a {
				e.writeString(s)
			}

		case []uint8:
			writeArray(e, a)
		case []int8:
			writeArray(e, a)
		case []uint16:
			writeArray(e, a)
		case []int16:
			writeArray(e, a)
		case []uint32:
			writeArray(e, a)
		case []int32:
			writeArray(e, a)
		case []float32:
			writeArray(e, a)
		case []bool:
			writeArray(e, a)
		case []uint64:
			writeArray(e, a)
		case []int64:
			writeArray(e, a)
		case []float64:
			writeArray(e, a)
		}

	default:
		e.write(v)
	}
}

// writeArray writes the length and elements of an array of fixed-size
// values.
func writeArray[T Scalar](e *encoder, a []T) {
	e.write(uint64(len(a)))

	// binary.Write handles slices of fixed-size types, bools included.
	e.write(a)
}
