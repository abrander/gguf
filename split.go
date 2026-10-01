package gguf

import (
	"fmt"
	"regexp"
)

// Large models are often split by llama.cpp's gguf-split in files named
// like model-00001-of-00004.gguf. The first file holds the metadata, and
// the tensors are spread over all of them.

// splitName matches the end of the name of a file that's part of a split
// model, like "-00001-of-00004.gguf".
var splitName = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// openSplit opens the rest of the files if r is the first file of a split
// model, and adds their tensors to r.Tensors. A later file opened on its
// own is left as it is, it's a valid GGUF file by itself.
func (r *Reader) openSplit() error {
	if _, found := r.Metadata["split.count"]; !found {
		return nil
	}

	count, err := MetaValueNumber[int](r.Metadata, "split.count")
	if err != nil {
		return err
	}

	if no, err := MetaValueNumber[int](r.Metadata, "split.no"); err == nil && no != 0 {
		return nil
	}

	if count <= 1 {
		return nil
	}

	if !splitName.MatchString(r.path) {
		return fmt.Errorf("%s is split in %d files, but its name doesn't end like -00001-of-%05d.gguf", r.path, count, count)
	}

	seen := make(map[string]bool, len(r.Tensors))
	for _, t := range r.Tensors {
		seen[t.Name] = true
	}

	for i := 1; i < count; i++ {
		path := splitName.ReplaceAllString(r.path, fmt.Sprintf("-%05d-of-%05d.gguf", i+1, count))

		part, err := openFile(path)
		if err != nil {
			return err
		}

		r.parts = append(r.parts, part)

		err = checkSplitValue(part.Metadata, "split.count", count)
		if err == nil {
			err = checkSplitValue(part.Metadata, "split.no", i)
		}

		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}

		for _, t := range part.Tensors {
			if seen[t.Name] {
				return fmt.Errorf("%s: tensor %q is in more than one file", path, t.Name)
			}

			seen[t.Name] = true
		}

		r.Tensors = append(r.Tensors, part.Tensors...)
	}

	err = checkSplitValue(r.Metadata, "split.tensors.count", len(r.Tensors))
	if err != nil {
		return fmt.Errorf("%s: %w", r.path, err)
	}

	return nil
}

// checkSplitValue checks that the metadata value name is want, if it's
// present.
func checkSplitValue(metadata Metadata, name string, want int) error {
	if _, found := metadata[name]; !found {
		return nil
	}

	got, err := MetaValueNumber[int](metadata, name)
	if err != nil {
		return err
	}

	if got != want {
		return fmt.Errorf("%s is %d, expected %d", name, got, want)
	}

	return nil
}
