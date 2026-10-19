package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// Types derived from the types in Number.
type (
	layers  int
	small   uint8
	ratio   float32
	counter uint64
)

func TestMetaValueNumberDerived(t *testing.T) {
	m := Metadata{}
	m.Set[uint32]("block_count", 32)
	m.Set[float32]("ratio", 0.5)
	m.Set("file_type", MostlyQ4_KM)
	m.Set[uint32]("general.file_type", 2)
	m.Set("text", "llama")

	n, err := MetaValueNumber[layers](m, "block_count")
	if err != nil || n != 32 {
		t.Fatalf("layers: got %v, %v", n, err)
	}

	s, err := MetaValueNumber[small](m, "block_count")
	if err != nil || s != 32 {
		t.Fatalf("small: got %v, %v", s, err)
	}

	r, err := MetaValueNumber[ratio](m, "ratio")
	if err != nil || r != 0.5 {
		t.Fatalf("ratio: got %v, %v", r, err)
	}

	c, err := MetaValueNumber[counter](m, "block_count")
	if err != nil || c != 32 {
		t.Fatalf("counter: got %v, %v", c, err)
	}

	// Filetypes are numbers too, whether stored as one or as a uint32.
	for _, name := range []string{"file_type", "general.file_type"} {
		i, err := m.Int(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		f, err := MetaValueNumber[Filetype](m, name)
		if err != nil || int(f) != i {
			t.Fatalf("%s: got %v and %v, %v", name, i, f, err)
		}
	}

	_, err = MetaValueNumber[layers](m, "text")
	if err == nil {
		t.Fatal("a string is not a number")
	}

	_, err = MetaValueNumber[layers](m, "missing")
	if err == nil {
		t.Fatal("missing value")
	}
}

func TestReadDerived(t *testing.T) {
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		var buf bytes.Buffer

		_ = binary.Write(&buf, order, uint8(7))
		_ = binary.Write(&buf, order, float32(1.25))
		_ = binary.Write(&buf, order, uint32(1000))

		s, err := read[small](&buf, order)
		if err != nil || s != 7 {
			t.Fatalf("%v small: got %v, %v", order, s, err)
		}

		r, err := read[ratio](&buf, order)
		if err != nil || r != 1.25 {
			t.Fatalf("%v ratio: got %v, %v", order, r, err)
		}

		c, err := readCast[uint32, counter](&buf, order)
		if err != nil || c != 1000 {
			t.Fatalf("%v counter: got %v, %v", order, c, err)
		}
	}
}
