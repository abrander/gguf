package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func testMetadata() Metadata {
	return Metadata{
		"general.architecture": "test",
		"general.alignment":    uint32(64),
		"general.file_type":    MostlyQ4_0,
		"test.uint8":           uint8(1),
		"test.int8":            int8(-2),
		"test.uint16":          uint16(3),
		"test.int16":           int16(-4),
		"test.uint32":          uint32(5),
		"test.int32":           int32(-6),
		"test.float32":         float32(7.5),
		"test.bool":            true,
		"test.string":          "hello, world",
		"test.uint64":          uint64(1 << 40),
		"test.int64":           int64(-1 << 40),
		"test.float64":         float64(-9.25),
		"test.array.uint8":     []uint8{1, 2, 3},
		"test.array.int8":      []int8{-1, 2},
		"test.array.uint16":    []uint16{4},
		"test.array.int16":     []int16{-5, 6},
		"test.array.uint32":    []uint32{7, 8, 9},
		"test.array.int32":     []int32{-10},
		"test.array.float32":   []float32{1.5, -2.5},
		"test.array.bool":      []bool{true, false, true},
		"test.array.string":    []string{"a", "", "ccc"},
		"test.array.uint64":    []uint64{1 << 50},
		"test.array.int64":     []int64{-1 << 50},
		"test.array.float64":   []float64{3.125},
		"test.array.empty":     []string{},
	}
}

func testTensors() []TensorInfo {
	return []TensorInfo{
		{Name: "f32", Type: GgmlF32, Dimensions: []uint64{4, 2}},
		{Name: "empty", Type: GgmlF32, Dimensions: []uint64{0}},
		{Name: "q4_0", Type: GgmlQ4_0, Dimensions: []uint64{64}},
		{Name: "i8", Type: GgmlI8, Dimensions: []uint64{3}},
	}
}

// testData returns some data for each tensor.
func testData(t *testing.T, tensors []TensorInfo) [][]byte {
	data := make([][]byte, len(tensors))

	for i, tensor := range tensors {
		size, err := tensorSize(tensor)
		if err != nil {
			t.Fatal(err)
		}

		data[i] = make([]byte, size)
		for j := range data[i] {
			data[i][j] = byte(i*31 + j)
		}
	}

	return data
}

// writeTest writes a GGUF file, with the tensors in reverse order.
func writeTest(t *testing.T, h Header, data [][]byte) []byte {
	var f memFile

	w, err := NewWriter(&f, h)
	if err != nil {
		t.Fatal(err)
	}

	for i := len(h.Tensors) - 1; i >= 0; i-- {
		err = w.WriteTensor(&h.Tensors[i], data[i])
		if err != nil {
			t.Fatal(err)
		}
	}

	err = w.Close()
	if err != nil {
		t.Fatal(err)
	}

	return f.b
}

// checkTensors checks that the tensors of r are tensors with data.
func checkTensors(t *testing.T, r *Reader, tensors []TensorInfo, data [][]byte) {
	t.Helper()

	if len(r.Tensors) != len(tensors) {
		t.Fatalf("got %d tensors, want %d", len(r.Tensors), len(tensors))
	}

	for i, got := range r.Tensors {
		want := tensors[i]

		if got.Name != want.Name || got.Type != want.Type || !reflect.DeepEqual(got.Dimensions, want.Dimensions) {
			t.Fatalf("tensor %d: got %s %v %v", i, got.Name, got.Type, got.Dimensions)
		}

		if len(data[i]) == 0 {
			continue
		}

		src, err := got.Reader()
		if err != nil {
			t.Fatal(err)
		}

		b := make([]byte, len(data[i]))

		_, err = io.ReadFull(src, b)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(b, data[i]) {
			t.Fatalf("tensor %q: data differs", got.Name)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		t.Run(order.String(), func(t *testing.T) {
			tensors := testTensors()
			data := testData(t, tensors)

			file := writeTest(t, Header{ByteOrder: order, Metadata: testMetadata(), Tensors: tensors}, data)

			r, err := Open(bytes.NewReader(file))
			if err != nil {
				t.Fatal(err)
			}

			if r.Version != 3 || r.ByteOrder != order {
				t.Fatalf("got version %d, %v", r.Version, r.ByteOrder)
			}

			if !reflect.DeepEqual(r.Metadata, testMetadata()) {
				t.Fatalf("metadata differs:\n got %#v\nwant %#v", r.Metadata, testMetadata())
			}

			checkTensors(t, r, tensors, data)

			for _, tensor := range r.Tensors {
				if tensor.DataOffset()%64 != 0 {
					t.Fatalf("tensor %q is at %d, not aligned to 64", tensor.Name, tensor.DataOffset())
				}
			}

			if int64(len(file))%64 != 0 {
				t.Fatalf("file size %d is not padded to 64", len(file))
			}
		})
	}
}

func TestConcurrent(t *testing.T) {
	var tensors []TensorInfo

	for i := 0; i < 100; i++ {
		tensors = append(tensors, TensorInfo{
			Name:       strings.Repeat("t", i+1),
			Type:       GgmlI8,
			Dimensions: []uint64{uint64(i + 1)},
		})
	}

	data := testData(t, tensors)

	var f memFile

	w, err := NewWriter(&f, Header{Tensors: tensors})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	errs := make(chan error, len(tensors))

	for i := range tensors {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			errs <- w.CopyTensor(&tensors[i], bytes.NewReader(data[i]))
		}(i)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}

	err = w.Close()
	if err != nil {
		t.Fatal(err)
	}

	r, err := Open(bytes.NewReader(f.b))
	if err != nil {
		t.Fatal(err)
	}

	checkTensors(t, r, tensors, data)
}

func TestFile(t *testing.T) {
	tensors := testTensors()
	data := testData(t, tensors)
	path := filepath.Join(t.TempDir(), "test.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	w, err := NewWriter(f, Header{Metadata: testMetadata(), Tensors: tensors})
	if err != nil {
		t.Fatal(err)
	}

	for i := range tensors {
		err = w.WriteTensor(&tensors[i], data[i])
		if err != nil {
			t.Fatal(err)
		}
	}

	err = w.Close()
	if err == nil {
		err = f.Close()
	}

	if err != nil {
		t.Fatal(err)
	}

	r, err := OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}

	defer r.Close()

	checkTensors(t, r, tensors, data)
}

func TestCopy(t *testing.T) {
	tensors := testTensors()
	file := writeTest(t, Header{Metadata: testMetadata(), Tensors: tensors}, testData(t, tensors))

	r, err := Open(bytes.NewReader(file))
	if err != nil {
		t.Fatal(err)
	}

	var f memFile

	err = Copy(&f, r, r.Metadata)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(f.b, file) {
		t.Fatal("copy of unchanged file differs")
	}

	meta := Metadata{"split.count": uint16(3), "general.name": "copied"}

	f = memFile{}

	err = Copy(&f, r, meta)
	if err != nil {
		t.Fatal(err)
	}

	c, err := Open(bytes.NewReader(f.b))
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(c.Metadata, Metadata{"general.name": "copied"}) {
		t.Fatalf("got metadata %v", c.Metadata)
	}

	if len(meta) != 2 {
		t.Fatal("Copy changed the metadata given")
	}
}

func TestTensorCopy(t *testing.T) {
	// Merge the tensors of two files into one.
	tensors := testTensors()
	data := testData(t, tensors)

	a, err := Open(bytes.NewReader(writeTest(t, Header{Tensors: tensors[:2]}, data[:2])))
	if err != nil {
		t.Fatal(err)
	}

	b, err := Open(bytes.NewReader(writeTest(t, Header{Tensors: tensors[2:]}, data[2:])))
	if err != nil {
		t.Fatal(err)
	}

	var f memFile

	w, err := NewWriter(&f, Header{Tensors: append(append([]TensorInfo{}, a.Tensors...), b.Tensors...)})
	if err != nil {
		t.Fatal(err)
	}

	for _, r := range []*Reader{b, a} {
		for i := range r.Tensors {
			err = TensorCopy(w, &r.Tensors[i])
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	err = w.Close()
	if err != nil {
		t.Fatal(err)
	}

	merged, err := Open(bytes.NewReader(f.b))
	if err != nil {
		t.Fatal(err)
	}

	checkTensors(t, merged, tensors, data)
}

func TestValue(t *testing.T) {
	m := Metadata{}

	ok := []struct {
		name string
		v    Value
		want interface{}
	}{
		{"uint32", Value{Type: Uint32, Data: uint32(5)}, uint32(5)},
		{"filetype", Value{Type: Uint32, Data: MostlyQ4_0}, uint32(MostlyQ4_0)},
		{"general.file_type", Value{Type: Uint32, Data: MostlyQ4_0}, MostlyQ4_0},
		{"general.file_type", Value{Type: Uint32, Data: uint32(7)}, MostlyQ8_0},
		{"array", Value{Type: Array, ElementType: Int16, Data: []int16{1, -2}}, []int16{1, -2}},
	}

	for _, c := range ok {
		err := m.SetValue(c.name, c.v)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if !reflect.DeepEqual(m[c.name], c.want) {
			t.Fatalf("%s: got %#v, want %#v", c.name, m[c.name], c.want)
		}
	}

	// There's no conversion, the Go type must match.
	bad := []Value{
		{Type: Uint32, Data: 5},
		{Type: Uint8, Data: uint32(5)},
		{Type: String, Data: nil},
		{Type: Array, ElementType: Uint8, Data: []int8{1}},
		{Type: Array, ElementType: Uint8, Data: uint8(1)},
		{Type: Uint8, Data: []uint8{1}},
		{Type: Array, ElementType: Uint8, Data: []int{1}},
	}

	for _, v := range bad {
		err := m.SetValue("bad", v)
		if err == nil {
			t.Fatalf("%+v: no error", v)
		}
	}

	if _, found := m["bad"]; found {
		t.Fatal("bad value was set")
	}

	// Merging values from one file into another.
	src := testMetadata()
	dst := Metadata{}

	for name := range src {
		v, err := src.Value(name)
		if err != nil {
			t.Fatal(err)
		}

		err = dst.SetValue(name, v)
		if err != nil {
			t.Fatal(err)
		}
	}

	if !reflect.DeepEqual(src, dst) {
		t.Fatalf("merged metadata differs:\n got %#v\nwant %#v", dst, src)
	}

	v, err := src.Value("test.array.float32")
	if err != nil {
		t.Fatal(err)
	}

	if v.Type != Array || v.ElementType != Float32 || v.String() != "array of float32" {
		t.Fatalf("got %v", v)
	}

	_, err = Metadata{"int": 1}.Value("int")
	if err == nil {
		t.Fatal("int has no GGUF type")
	}
}

// Types derived from the types in Scalar.
type (
	flag      bool
	label     string
	tokenType int32
)

func TestSet(t *testing.T) {
	m := Metadata{}

	m.Set("string", "llama")
	m.Set("bool", true)
	m.Set[uint32]("constant", 4096)
	m.Set[uint32]("general.file_type", 2)
	m.SetArray[int32]("array", []int32{1, -1, 3})
	m.SetArray("strings", []string{"a", "b"})
	m.SetArray[uint8]("empty", nil)

	// Derived types are stored as the type they're derived from.
	m.Set("filetype", MostlyQ4_0)
	m.Set("derived.counter", counter(64))
	m.Set("derived.small", small(7))
	m.Set("derived.ratio", ratio(0.5))
	m.Set("derived.flag", flag(true))
	m.Set("derived.label", label("llama"))
	m.SetArray("derived.array", []tokenType{1, -1, 3})
	m.SetArray("derived.strings", []label{"a", "b"})
	m.SetArray[tokenType]("derived.empty", nil)

	want := Metadata{
		"string":            "llama",
		"bool":              true,
		"constant":          uint32(4096),
		"general.file_type": MostlyQ4_0,
		"array":             []int32{1, -1, 3},
		"strings":           []string{"a", "b"},
		"empty":             []uint8(nil),
		"filetype":          uint32(MostlyQ4_0),
		"derived.counter":   uint64(64),
		"derived.small":     uint8(7),
		"derived.ratio":     float32(0.5),
		"derived.flag":      true,
		"derived.label":     "llama",
		"derived.array":     []int32{1, -1, 3},
		"derived.strings":   []string{"a", "b"},
		"derived.empty":     []int32(nil),
	}

	if !reflect.DeepEqual(m, want) {
		t.Fatalf("got %#v\nwant %#v", m, want)
	}

	// SetArray copies, whether the type is derived or not.
	ints := []int32{1, 2}
	labels := []label{"a"}

	m.SetArray("copy.ints", ints)
	m.SetArray("copy.labels", labels)
	m.SetArray("copy.empty", []int32{})

	ints[0], labels[0] = 9, "z"

	if m["copy.ints"].([]int32)[0] != 1 || m["copy.labels"].([]string)[0] != "a" {
		t.Fatal("SetArray didn't copy")
	}

	if a := m["copy.empty"].([]int32); a == nil {
		t.Fatal("empty slice became nil")
	}

	for _, k := range []string{"copy.ints", "copy.labels", "copy.empty"} {
		delete(m, k)
	}

	// Everything Set and SetArray store can be written and read back.
	var f memFile

	w, err := NewWriter(&f, Header{Metadata: m})
	if err != nil {
		t.Fatal(err)
	}

	err = w.Close()
	if err != nil {
		t.Fatal(err)
	}

	r, err := Open(bytes.NewReader(f.b))
	if err != nil {
		t.Fatal(err)
	}

	if len(r.Metadata) != len(want) {
		t.Fatalf("got %d values, want %d", len(r.Metadata), len(want))
	}

	if a, _ := MetaValue[[]int32](r.Metadata, "array"); !reflect.DeepEqual(a, []int32{1, -1, 3}) {
		t.Fatalf("got %v", a)
	}
}

func TestHeaderErrors(t *testing.T) {
	headers := map[string]Header{
		"int":         {Metadata: Metadata{"x": 1}},
		"nested":      {Metadata: Metadata{"x": [][]uint8{{1}}}},
		"alignment":   {Metadata: Metadata{"general.alignment": uint32(48)}},
		"alignment-0": {Metadata: Metadata{"general.alignment": uint32(0)}},
		"alignment-t": {Metadata: Metadata{"general.alignment": uint64(32)}},
		"duplicate":   {Tensors: []TensorInfo{{Name: "a", Type: GgmlF32}, {Name: "a", Type: GgmlF32}}},
		"row":         {Tensors: []TensorInfo{{Name: "a", Type: GgmlQ4_0, Dimensions: []uint64{16, 2}}}},
		"type":        {Tensors: []TensorInfo{{Name: "a", Type: GgmlQ4_2, Dimensions: []uint64{32}}}},
		"wraps":       {Tensors: []TensorInfo{{Name: "a", Type: GgmlI8, Dimensions: []uint64{1 << 62, 4}}}},
		"too-large":   {Tensors: []TensorInfo{{Name: "a", Type: GgmlI8, Dimensions: []uint64{1 << 32, 1 << 31}}}},
	}

	for name, h := range headers {
		var f memFile

		_, err := NewWriter(&f, h)
		if err == nil {
			t.Fatalf("%s: no error", name)
		}

		if len(f.b) != 0 {
			t.Fatalf("%s: wrote a partial header", name)
		}
	}
}

func TestTensorErrors(t *testing.T) {
	h := Header{Tensors: []TensorInfo{
		{Name: "a", Type: GgmlI8, Dimensions: []uint64{4}},
		{Name: "b", Type: GgmlI8, Dimensions: []uint64{4}},
		{Name: "c", Type: GgmlI8, Dimensions: []uint64{4}},
	}}

	w, err := NewWriter(&memFile{}, h)
	if err != nil {
		t.Fatal(err)
	}

	mustFail := func(what string, err error, contains string) {
		t.Helper()

		if err == nil || !strings.Contains(err.Error(), contains) {
			t.Fatalf("%s: got %v", what, err)
		}
	}

	a, _ := w.TensorInfo("a")
	b, _ := w.TensorInfo("b")
	c, _ := w.TensorInfo("c")

	_, err = w.TensorInfo("x")
	mustFail("lookup", err, "not found")

	mustFail("nil", w.WriteTensor(nil, nil), "no tensor")
	mustFail("unknown", w.WriteTensor(&TensorInfo{Name: "x", Type: GgmlI8, Dimensions: []uint64{4}},
		[]byte{1, 2, 3, 4}), "not in the header")
	mustFail("type", w.WriteTensor(&TensorInfo{Name: "a", Type: GgmlI32, Dimensions: []uint64{1}},
		[]byte{1, 2, 3, 4}), "is I8 [4] in the header, got I32 [1]")
	mustFail("shape", w.WriteTensor(&TensorInfo{Name: "a", Type: GgmlI8, Dimensions: []uint64{2, 2}},
		[]byte{1, 2, 3, 4}), "is I8 [4] in the header, got I8 [2 2]")
	mustFail("short", w.WriteTensor(a, []byte{1, 2, 3}), "is 4 bytes, got 3")
	mustFail("long", w.WriteTensor(a, []byte{1, 2, 3, 4, 5}), "is 4 bytes, got 5")

	err = w.WriteTensor(a, []byte{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}

	mustFail("twice", w.WriteTensor(a, []byte{1, 2, 3, 4}), "written already")
	mustFail("no reader", TensorCopy(w, c), "not from a Reader")

	err = w.CopyTensor(b, bytes.NewReader([]byte{1, 2}))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("short reader: got %v", err)
	}

	// A failed tensor can be tried again.
	err = w.CopyTensor(b, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6}))
	if err != nil {
		t.Fatal(err)
	}

	mustFail("close", w.Close(), `1 tensors are not written: c`)
	mustFail("after close", w.WriteTensor(c, []byte{1, 2, 3, 4}), "closed")
}

func TestTensorCopyMismatch(t *testing.T) {
	// The source has tensors with the names of the destination, but
	// other shapes and types. Matching by name alone would copy too
	// little, or read into the next tensor.
	src := []TensorInfo{
		{Name: "bigger", Type: GgmlI8, Dimensions: []uint64{8}},
		{Name: "smaller", Type: GgmlI8, Dimensions: []uint64{2}},
		{Name: "type", Type: GgmlI32, Dimensions: []uint64{1}},
	}

	r, err := Open(bytes.NewReader(writeTest(t, Header{Tensors: src}, testData(t, src))))
	if err != nil {
		t.Fatal(err)
	}

	w, err := NewWriter(&memFile{}, Header{Tensors: []TensorInfo{
		{Name: "bigger", Type: GgmlI8, Dimensions: []uint64{4}},
		{Name: "smaller", Type: GgmlI8, Dimensions: []uint64{4}},
		{Name: "type", Type: GgmlI8, Dimensions: []uint64{4}},
	}})
	if err != nil {
		t.Fatal(err)
	}

	for i := range r.Tensors {
		err = TensorCopy(w, &r.Tensors[i])
		if err == nil || !strings.Contains(err.Error(), "in the header, got") {
			t.Fatalf("%s: got %v", r.Tensors[i].Name, err)
		}
	}
}
