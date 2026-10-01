//go:build linux

package gguf

import (
	"os"
	"syscall"
	"unsafe"
)

// Advice tells the operating system how Data will be used.
type Advice uintptr

const (
	AdviceNormal     = Advice(syscall.MADV_NORMAL)
	AdviceRandom     = Advice(syscall.MADV_RANDOM)
	AdviceSequential = Advice(syscall.MADV_SEQUENTIAL)
	AdviceWillNeed   = Advice(syscall.MADV_WILLNEED)
	AdviceDontNeed   = Advice(syscall.MADV_DONTNEED)
)

var pageSize = uintptr(os.Getpagesize())

// pages returns the start and length of the whole pages covering d. The
// mappings start on a page boundary, so the pages are always inside the
// mapping.
func (d Data) pages() (uintptr, uintptr) {
	addr := uintptr(unsafe.Pointer(&d[0]))
	start := addr &^ (pageSize - 1)

	return start, addr + uintptr(len(d)) - start
}

// pageSyscall calls a syscall taking an address and a length, and maybe a
// third argument, for the pages covering d.
func (d Data) pageSyscall(trap uintptr, arg uintptr) error {
	if len(d) == 0 {
		return nil
	}

	start, length := d.pages()

	_, _, errno := syscall.Syscall(trap, start, length, arg)
	if errno != 0 {
		return errno
	}

	return nil
}

func (d Data) madvise(advice Advice) error {
	return d.pageSyscall(syscall.SYS_MADVISE, uintptr(advice))
}

func (d Data) mlock() error {
	return d.pageSyscall(syscall.SYS_MLOCK, 0)
}

func (d Data) munlock() error {
	return d.pageSyscall(syscall.SYS_MUNLOCK, 0)
}
