package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"sort"
	"strings"
	"sync"
)

// ErrWriterClosed is returned when an operation is attempted on a closed
// Writer.
var ErrWriterClosed = errors.New("gguf: writer is closed")

// Header describes a GGUF file to write.
type Header struct {
	// ByteOrder is the byte order of the header. Tensor data is written
	// as it is. Little-endian is used if it's nil.
	ByteOrder binary.ByteOrder

	// Metadata is written with the keys sorted, general.architecture
	// first. Values must be of the Go types Open returns, which
	// Metadata.Set and Metadata.SetArray make sure of.
	Metadata Metadata

	// Tensors are the tensors, in the order they're placed in the file.
	// Offset is ignored, the Writer computes it.
	Tensors []TensorInfo
}

type tensorState int

const (
	unwritten tensorState = iota
	writing
	written
)

// Writer writes a GGUF version 3 file. NewWriter writes the header, which
// fixes where the data of every tensor goes. The tensors can then be
// written in any order, and from several goroutines at once.
type Writer struct {
	w         io.WriterAt
	alignment int64
	dataStart int64

	tensors []TensorInfo
	index   map[string]int
	sizes   []int64
	offsets []int64

	mu     sync.Mutex
	state  []tensorState
	closed bool
}

// NewWriter writes the header h to w, and returns a Writer for the
// tensor data.
func NewWriter(w io.WriterAt, h Header) (*Writer, error) {
	order := h.ByteOrder
	if order == nil {
		order = binary.LittleEndian
	}

	alignment := defaultAlignment

	if a, found := h.Metadata["general.alignment"]; found {
		v, ok := a.(uint32)
		if !ok || v == 0 || v&(v-1) != 0 {
			return nil, fmt.Errorf("general.alignment must be a uint32 power of two, got %v (%T)", a, a)
		}

		alignment = int64(v)
	}

	gw := &Writer{
		w:         w,
		alignment: alignment,
		tensors:   append([]TensorInfo(nil), h.Tensors...),
		index:     make(map[string]int, len(h.Tensors)),
		sizes:     make([]int64, len(h.Tensors)),
		offsets:   make([]int64, len(h.Tensors)),
		state:     make([]tensorState, len(h.Tensors)),
	}

	var buf bytes.Buffer

	e := &encoder{w: &buf, order: order}

	_, _ = buf.WriteString(magic)
	e.write(uint32(3))
	e.write(uint64(len(h.Tensors)))
	e.write(uint64(len(h.Metadata)))

	for _, key := range sortedKeys(h.Metadata) {
		e.writeString(key)
		e.writeValue(h.Metadata[key])

		if e.err != nil {
			return nil, fmt.Errorf("metadata value %q: %w", key, e.err)
		}
	}

	offset := int64(0)

	for i, t := range h.Tensors {
		if _, found := gw.index[t.Name]; found {
			return nil, fmt.Errorf("tensor %q is there more than once", t.Name)
		}

		gw.index[t.Name] = i

		size, err := tensorSize(t)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
		}

		gw.sizes[i] = size
		gw.offsets[i] = offset

		e.writeString(t.Name)
		e.write(uint32(len(t.Dimensions)))
		e.write(t.Dimensions)
		e.write(uint32(t.Type))
		e.write(uint64(offset))

		offset = gw.align(offset + size)
	}

	if e.err != nil {
		return nil, e.err
	}

	gw.dataStart = gw.align(int64(buf.Len()))
	_, _ = buf.Write(make([]byte, gw.dataStart-int64(buf.Len())))

	_, err := w.WriteAt(buf.Bytes(), 0)
	if err != nil {
		return nil, err
	}

	return gw, nil
}

// sortedKeys returns the keys of m sorted, with general.architecture
// first, like other tools write it.
func sortedKeys(m Metadata) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Slice(keys, func(i, j int) bool {
		ai, aj := keys[i] == "general.architecture", keys[j] == "general.architecture"
		if ai != aj {
			return ai
		}

		return keys[i] < keys[j]
	})

	return keys
}

// valueCount returns the number of values in a tensor with the given
// dimensions. It's an error if it doesn't fit in an int64.
func valueCount(dimensions []uint64) (int64, error) {
	values := uint64(1)

	for _, d := range dimensions {
		hi, lo := bits.Mul64(values, d)
		if hi != 0 || lo > math.MaxInt64 {
			return 0, fmt.Errorf("dimensions %v are too large", dimensions)
		}

		values = lo
	}

	return int64(values), nil
}

// tensorSize returns the size of the data of t in bytes.
func tensorSize(t TensorInfo) (int64, error) {
	values, err := valueCount(t.Dimensions)
	if err != nil {
		return 0, err
	}

	size, err := t.Type.ByteSize(values)
	if err != nil {
		return 0, err
	}

	// ggml stores rows as whole blocks.
	if len(t.Dimensions) > 0 && t.Dimensions[0]%uint64(t.Type.BlockValues()) != 0 {
		return 0, fmt.Errorf("first dimension %d is not a multiple of the %v block size %d",
			t.Dimensions[0], t.Type, t.Type.BlockValues())
	}

	return size, nil
}

func (w *Writer) align(n int64) int64 {
	return (n + w.alignment - 1) / w.alignment * w.alignment
}

// TensorInfo returns the tensor info for the tensor with the given
// name. If the tensor is not in the header, an error is returned.
func (w *Writer) TensorInfo(name string) (*TensorInfo, error) {
	i, found := w.index[name]
	if !found {
		return nil, fmt.Errorf("tensor %q not found", name)
	}

	return &w.tensors[i], nil
}

// begin claims the tensor t for writing. It's found by name, and its type
// and dimensions must match the header. size is checked unless it's
// negative.
func (w *Writer) begin(t *TensorInfo, size int64) (int, error) {
	if t == nil {
		return 0, errors.New("gguf: no tensor")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return 0, ErrWriterClosed
	}

	i, found := w.index[t.Name]
	if !found {
		return 0, fmt.Errorf("gguf: tensor %q is not in the header", t.Name)
	}

	want := &w.tensors[i]

	if t.Type != want.Type || !sameDimensions(t.Dimensions, want.Dimensions) {
		return 0, fmt.Errorf("gguf: tensor %q is %v %v in the header, got %v %v",
			t.Name, want.Type, want.Dimensions, t.Type, t.Dimensions)
	}

	if size >= 0 && size != w.sizes[i] {
		return 0, fmt.Errorf("gguf: tensor %q is %d bytes, got %d", t.Name, w.sizes[i], size)
	}

	switch w.state[i] {
	case writing:
		return 0, fmt.Errorf("gguf: tensor %q is being written already", t.Name)

	case written:
		return 0, fmt.Errorf("gguf: tensor %q is written already", t.Name)
	}

	w.state[i] = writing

	return i, nil
}

func sameDimensions(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// finish marks tensor i as written, or as unwritten if err is set, so it
// can be tried again.
func (w *Writer) finish(i int, err error) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err != nil {
		w.state[i] = unwritten

		return fmt.Errorf("gguf: tensor %q: %w", w.tensors[i].Name, err)
	}

	w.state[i] = written

	return nil
}

// pad writes the padding after tensor i, so the file has the right
// length whichever tensor is written last.
func (w *Writer) pad(i int) error {
	end := w.dataStart + w.offsets[i] + w.sizes[i]

	_, err := w.w.WriteAt(make([]byte, w.align(end)-end), end)

	return err
}

// WriteTensor writes the data of the tensor t. t is found by name, and
// its type and dimensions must match the header, but it can come from
// anywhere, like another Reader. data must be exactly the size of the
// tensor. It's safe to call from several goroutines.
func (w *Writer) WriteTensor(t *TensorInfo, data []byte) error {
	i, err := w.begin(t, int64(len(data)))
	if err != nil {
		return err
	}

	_, err = w.w.WriteAt(data, w.dataStart+w.offsets[i])
	if err == nil {
		err = w.pad(i)
	}

	return w.finish(i, err)
}

// CopyTensor writes the data of the tensor t, read from r. t is matched
// like in WriteTensor. Exactly the size of the tensor is read, it's an
// error if r ends before. It's safe to call from several goroutines.
func (w *Writer) CopyTensor(t *TensorInfo, r io.Reader) error {
	i, err := w.begin(t, -1)
	if err != nil {
		return err
	}

	dst := io.NewOffsetWriter(w.w, w.dataStart+w.offsets[i])

	n, err := io.CopyBuffer(dst, io.LimitReader(r, w.sizes[i]), make([]byte, 1<<20))
	if err == nil && n != w.sizes[i] {
		err = fmt.Errorf("got %d of %d bytes: %w", n, w.sizes[i], io.ErrUnexpectedEOF)
	}

	if err == nil {
		err = w.pad(i)
	}

	return w.finish(i, err)
}

// Close checks that the data of all tensors has been written. It doesn't
// close the underlying io.WriterAt. Until Close returns nil the file is
// incomplete, but it looks valid, with zeros for the missing data. It's
// a good idea to write to a temporary file and rename it after Close.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrWriterClosed
	}

	w.closed = true

	var missing []string

	for i, state := range w.state {
		// Tensors without data are fine either way.
		if state != written && w.sizes[i] > 0 {
			missing = append(missing, w.tensors[i].Name)
		}
	}

	if len(missing) == 0 {
		return nil
	}

	count := len(missing)
	if count > 5 {
		missing = append(missing[:5], "...")
	}

	return fmt.Errorf("gguf: %d tensors are not written: %s", count, strings.Join(missing, ", "))
}

// TensorCopy copies the data of the tensor t from the Reader it belongs
// to, to the tensor with the same name in dst. The type and dimensions
// must match. It's safe to call from several goroutines if the Reader
// was opened from an io.ReaderAt, like an *os.File.
func TensorCopy(dst *Writer, t *TensorInfo) error {
	src, err := t.Reader()
	if err != nil {
		return fmt.Errorf("gguf: tensor %q: %w", t.Name, err)
	}

	return dst.CopyTensor(t, src)
}

// Copy writes a GGUF file to dst with the metadata m and the tensors of
// r, including the tensors of the other files of a split model. m is
// often r.Metadata, maybe with some changes. Split metadata, split.*,
// is left out, as the result is a single file. r must use the byte order
// of the output, tensor data isn't swapped.
func Copy(dst io.WriterAt, r *Reader, m Metadata) error {
	meta := make(Metadata, len(m))

	for k, v := range m {
		if !strings.HasPrefix(k, "split.") {
			meta[k] = v
		}
	}

	w, err := NewWriter(dst, Header{
		ByteOrder: r.ByteOrder,
		Metadata:  meta,
		Tensors:   r.Tensors,
	})
	if err != nil {
		return err
	}

	for i := range r.Tensors {
		err = TensorCopy(w, &r.Tensors[i])
		if err != nil {
			return err
		}
	}

	return w.Close()
}
