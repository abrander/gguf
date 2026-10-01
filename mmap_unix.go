//go:build unix

package gguf

import (
	"fmt"
	"os"
	"syscall"
)

// mapFile maps all of f into memory, read only. The mapping stays valid
// after f is closed.
func mapFile(f *os.File) ([]byte, error) {
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := info.Size()
	if size == 0 {
		return nil, nil
	}

	if int64(int(size)) != size {
		return nil, fmt.Errorf("file is too large to map (%d bytes)", size)
	}

	return syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
}

func unmapFile(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	return syscall.Munmap(data)
}
