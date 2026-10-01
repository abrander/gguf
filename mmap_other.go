//go:build !unix

package gguf

import (
	"os"
)

func mapFile(*os.File) ([]byte, error) {
	return nil, ErrUnsupported
}

func unmapFile([]byte) error {
	return nil
}
