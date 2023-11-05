package gguf

import (
	"fmt"
)

// Type is the type of a GGUF metadata value.
type Type uint32

const (
	Uint8   Type = 0
	Int8    Type = 1
	Uint16  Type = 2
	Int16   Type = 3
	Uint32  Type = 4
	Int32   Type = 5
	Float32 Type = 6
	Bool    Type = 7
	String  Type = 8
	Array   Type = 9

	// Added in v2
	Uint64  Type = 10
	Int64   Type = 11
	Float64 Type = 12
)

// String returns the string representation of a GGUF type.
// Implements fmt.Stringer.
func (t Type) String() string {
	switch t {
	case Uint8:
		return "uint8"

	case Int8:
		return "int8"

	case Uint16:
		return "uint16"

	case Int16:
		return "int16"

	case Uint32:
		return "uint32"

	case Int32:
		return "int32"

	case Float32:
		return "float32"

	case Bool:
		return "bool"

	case String:
		return "string"

	case Array:
		return "array"

	case Uint64:
		return "uint64"

	case Int64:
		return "int64"

	case Float64:
		return "float64"

	default:
		return fmt.Sprintf("unknown-type-%d", t)
	}
}
