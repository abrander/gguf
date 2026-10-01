package gguf

import (
	"errors"
)

// Data is the data of a tensor, as a slice of a read-only memory mapping
// of the file. It must not be modified, and it's invalid after the Reader
// is closed.
type Data []byte

// ErrUnsupported is returned on systems where the operation isn't
// implemented.
var ErrUnsupported = errors.New("not supported on this platform")

// Madvise tells the operating system how d will be used. The advice
// applies to whole pages, so it can include the start and end of the
// neighbouring tensors. It's only supported on Linux.
func (d Data) Madvise(advice Advice) error {
	return d.madvise(advice)
}

// Mlock locks d in memory, so it's never paged out. The lock applies to
// whole pages, so it can include the start and end of the neighbouring
// tensors. It's only supported on Linux.
func (d Data) Mlock() error {
	return d.mlock()
}

// Munlock undoes Mlock. It's only supported on Linux.
func (d Data) Munlock() error {
	return d.munlock()
}
