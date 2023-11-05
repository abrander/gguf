package gguf

import (
	"fmt"
)

// Metadata is a container for metadata in a GGUF file. Values are
// mapped to their corresponding Go types.
type Metadata map[string]interface{}

// Int returns the value of the metadata with the given name as an
// int. If the value cannot be represented as an int, an error is
// returned.
func (m Metadata) Int(name string) (int, error) {
	return MetaValueNumber[int](m, name)
}

// Any returns the value of the metadata with the given name as an
// interface{}.
func (m Metadata) Any(name string) (interface{}, error) {
	return MetaValue[any](m, name)
}

// String returns the value of the metadata with the given name as
// a string. If the value is not a string, an error is returned.
func (m Metadata) String(name string) (string, error) {
	return MetaValue[string](m, name)
}

// MetaValue returns the value of the metadata with the given name as
// type T. If the value is not a T, an error is returned.
func MetaValue[T any](metadata Metadata, name string) (T, error) {
	var zero T
	v, found := metadata[name]
	if !found {
		return zero, fmt.Errorf("metadata value %q not found", name)
	}

	if _, ok := v.(T); !ok {
		return zero, fmt.Errorf("metadata value %q is not of type %T, type is %T", name, zero, v)
	}

	return v.(T), nil
}

// MetaValueNumber returns the value of the metadata with the given
// name as a number. If the value is not a number, an error is
// returned. The number will be cast to the type T. This can be
// useful if you don't really care about the exact type of the
// number.
func MetaValueNumber[T ~int | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64](metadata Metadata, name string) (T, error) {
	v, found := metadata[name]
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
