package gguf

import (
	"reflect"
	"slices"
)

// Reflection is only used to store values of types derived from the GGUF
// Go types, like `type layers uint32`, as the type they're derived from.
// A type switch can't see what a type is derived from. The constraints
// make sure there always is a GGUF Go type to convert to.

// exactTypes maps the kinds of the types in Scalar to the GGUF Go types.
var exactTypes = map[reflect.Kind]reflect.Type{
	reflect.Uint8:   reflect.TypeFor[uint8](),
	reflect.Int8:    reflect.TypeFor[int8](),
	reflect.Uint16:  reflect.TypeFor[uint16](),
	reflect.Int16:   reflect.TypeFor[int16](),
	reflect.Uint32:  reflect.TypeFor[uint32](),
	reflect.Int32:   reflect.TypeFor[int32](),
	reflect.Float32: reflect.TypeFor[float32](),
	reflect.Bool:    reflect.TypeFor[bool](),
	reflect.String:  reflect.TypeFor[string](),
	reflect.Uint64:  reflect.TypeFor[uint64](),
	reflect.Int64:   reflect.TypeFor[int64](),
	reflect.Float64: reflect.TypeFor[float64](),
}

// exactType returns the GGUF Go type T is derived from, or T itself.
func exactType[T Scalar]() reflect.Type {
	return exactTypes[reflect.TypeFor[T]().Kind()]
}

// exact returns v as the GGUF Go type it's derived from.
func exact[T Scalar](v T) interface{} {
	return reflect.ValueOf(v).Convert(exactType[T]()).Interface()
}

// exactSlice returns a copy of a, as a slice of the GGUF Go type T is
// derived from. A nil slice stays nil.
func exactSlice[T Scalar](a []T) interface{} {
	t := exactType[T]()

	// Arrays can be large, like a tokenizer's vocabulary, so the common
	// case is copied without reflection.
	if t == reflect.TypeFor[T]() {
		return slices.Clone(a)
	}

	if a == nil {
		return reflect.Zero(reflect.SliceOf(t)).Interface()
	}

	out := reflect.MakeSlice(reflect.SliceOf(t), len(a), len(a))

	for i, v := range a {
		out.Index(i).Set(reflect.ValueOf(v).Convert(t))
	}

	return out.Interface()
}
