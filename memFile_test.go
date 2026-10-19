package gguf

import (
	"io"
	"sync"
)

// memFile is an in-memory io.WriterAt.
type memFile struct {
	mu sync.Mutex
	b  []byte
}

var _ io.WriterAt = (*memFile)(nil)

func (f *memFile) WriteAt(p []byte, off int64) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if end := int(off) + len(p); end > len(f.b) {
		f.b = append(f.b, make([]byte, end-len(f.b))...)
	}

	return copy(f.b[off:], p), nil
}
