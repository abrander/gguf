package gguf

// Filetype is the type of the majority of the tensors in the file.
type Filetype uint32

const (
	AllF32            Filetype = 0
	MostlyF16         Filetype = 1
	MostlyQ4_0        Filetype = 2
	MostlyQ4_1        Filetype = 3
	MostlyQ4_1SomeF16 Filetype = 4
	MostlyQ4_2        Filetype = 5 // support removed from llama.cpp/ggml
	MostlyQ4_3        Filetype = 6 // support removed from llama.cpp/ggml
	MostlyQ8_0        Filetype = 7
	MostlyQ5_0        Filetype = 8
	MostlyQ5_1        Filetype = 9
	MostlyQ2_K        Filetype = 10
	MostlyQ3_KS       Filetype = 11
	MostlyQ3_KM       Filetype = 12
	MostlyQ3_KL       Filetype = 13
	MostlyQ4_KS       Filetype = 14
	MostlyQ4_KM       Filetype = 15
	MostlyQ5_KS       Filetype = 16
	MostlyQ5_KM       Filetype = 17
	MostlyQ6_K        Filetype = 18
)

var ftypeNames = map[Filetype]string{
	AllF32:            "all F32",
	MostlyF16:         "mostly F16",
	MostlyQ4_0:        "mostly Q4_0",
	MostlyQ4_1:        "mostly Q4_1",
	MostlyQ4_1SomeF16: "mostly Q4_1, some F16",
	MostlyQ4_2:        "mostly Q4_2",
	MostlyQ4_3:        "mostly Q4_3",
	MostlyQ8_0:        "mostly Q8_0",
	MostlyQ5_0:        "mostly Q5_0",
	MostlyQ5_1:        "mostly Q5_1",
	MostlyQ2_K:        "mostly Q2_K",
	MostlyQ3_KS:       "mostly Q3_K - Small",
	MostlyQ3_KM:       "mostly Q3_K - Medium",
	MostlyQ3_KL:       "mostly Q3_K - Large",
	MostlyQ4_KS:       "mostly Q4_K - Small",
	MostlyQ4_KM:       "mostly Q4_K - Medium",
	MostlyQ5_KS:       "mostly Q5_K - Small",
	MostlyQ5_KM:       "mostly Q5_K - Medium",
	MostlyQ6_K:        "mostly Q6_K",
}

// String return a string representation of the Filetype. All strings are
// matched to those used in llama.cpp.
func (f Filetype) String() string {
	name, found := ftypeNames[f]
	if found {
		return name
	}

	return "UNKNOWN"
}
