//go:build !linux

package gguf

// Advice tells the operating system how Data will be used.
type Advice uintptr

const (
	AdviceNormal Advice = iota
	AdviceRandom
	AdviceSequential
	AdviceWillNeed
	AdviceDontNeed
)

func (Data) madvise(Advice) error {
	return ErrUnsupported
}

func (Data) mlock() error {
	return ErrUnsupported
}

func (Data) munlock() error {
	return ErrUnsupported
}
