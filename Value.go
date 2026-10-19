package gguf

import (
	"fmt"
)

// Scalar is the GGUF value types, and types derived from them. Values of
// derived types are stored as the GGUF type they're derived from.
type Scalar interface {
	Number | ~bool | ~string
}

// Value is a metadata value together with its GGUF type.
type Value struct {
	// Type is the type of the value. For arrays it's Array, and
	// ElementType is the type of the elements.
	Type        Type
	ElementType Type

	// Data is the value as the Go type matching Type, like uint32 for
	// Uint32 or []string for an array of String.
	Data interface{}
}

// Value returns the metadata value with the given name, together with
// its GGUF type.
func (m Metadata) Value(name string) (Value, error) {
	v, found := m[name]
	if !found {
		return Value{}, fmt.Errorf("metadata value %q not found", name)
	}

	typ, elem, err := typeOf(v)
	if err != nil {
		return Value{}, fmt.Errorf("metadata value %q: %w", name, err)
	}

	return Value{Type: typ, ElementType: elem, Data: v}, nil
}

// Set sets the metadata value with the given name. The GGUF type follows
// from T, or the type T is derived from.
//
//	m.Set("general.architecture", "llama")
//	m.Set[uint32]("llama.context_length", 4096)
func (m Metadata) Set[T Scalar](name string, v T) {
	m.set(name, exact(v))
}

// SetArray sets the metadata value with the given name to an array of T,
// or the type T is derived from. v is copied.
//
//	m.SetArray[int32]("tokenizer.ggml.token_type", []int32{1, 1, 3})
func (m Metadata) SetArray[T Scalar](name string, v []T) {
	m[name] = exactSlice(v)
}

// SetValue sets the metadata value with the given name, like one returned
// by Value. v.Data must be the Go type matching v.Type, there's no
// conversion.
func (m Metadata) SetValue(name string, v Value) error {
	typ, elem, err := typeOf(v.Data)
	if err != nil {
		return fmt.Errorf("metadata value %q: %w", name, err)
	}

	if typ != v.Type || (typ == Array && elem != v.ElementType) {
		return fmt.Errorf("metadata value %q: %T doesn't match %v", name, v.Data, v)
	}

	m.set(name, v.Data)

	return nil
}

// set sets a value of a known type. Like Open, general.file_type is
// stored as a Filetype, and anything else as a uint32.
func (m Metadata) set(name string, v interface{}) {
	switch x := v.(type) {
	case uint32:
		if name == "general.file_type" {
			v = Filetype(x)
		}

	case Filetype:
		if name != "general.file_type" {
			v = uint32(x)
		}
	}

	m[name] = v
}

// String returns the GGUF type of v, like "uint32" or "array of string".
func (v Value) String() string {
	if v.Type == Array {
		return "array of " + v.ElementType.String()
	}

	return v.Type.String()
}

// typeOf returns the GGUF type of the Go value v, and the element type if
// v is an array. Only the exact Go types used in Metadata, and Filetype,
// have a GGUF type.
func typeOf(v interface{}) (Type, Type, error) {
	switch v.(type) {
	case uint8:
		return Uint8, 0, nil
	case int8:
		return Int8, 0, nil
	case uint16:
		return Uint16, 0, nil
	case int16:
		return Int16, 0, nil
	case uint32, Filetype:
		return Uint32, 0, nil
	case int32:
		return Int32, 0, nil
	case float32:
		return Float32, 0, nil
	case bool:
		return Bool, 0, nil
	case string:
		return String, 0, nil
	case uint64:
		return Uint64, 0, nil
	case int64:
		return Int64, 0, nil
	case float64:
		return Float64, 0, nil

	case []uint8:
		return Array, Uint8, nil
	case []int8:
		return Array, Int8, nil
	case []uint16:
		return Array, Uint16, nil
	case []int16:
		return Array, Int16, nil
	case []uint32:
		return Array, Uint32, nil
	case []int32:
		return Array, Int32, nil
	case []float32:
		return Array, Float32, nil
	case []bool:
		return Array, Bool, nil
	case []string:
		return Array, String, nil
	case []uint64:
		return Array, Uint64, nil
	case []int64:
		return Array, Int64, nil
	case []float64:
		return Array, Float64, nil

	default:
		return 0, 0, fmt.Errorf("%T has no GGUF type", v)
	}
}
