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

// String return a string representation of the Filetype. All strings are
// matched to those used in llama.cpp.
func (f Filetype) String() string {
	switch f {
	case AllF32:
		return "ALL_F32"

	case MostlyF16:
		return "MOSTLY_F16"

	case MostlyQ4_0:
		return "MOSTLY_Q4_0"

	case MostlyQ4_1:
		return "MOSTLY_Q4_1"

	case MostlyQ4_1SomeF16:
		return "MOSTLY_Q4_1_SOME_F16"

	case MostlyQ4_2:
		return "MOSTLY_Q4_2"

	case MostlyQ4_3:
		return "MOSTLY_Q4_3"

	case MostlyQ8_0:
		return "MOSTLY_Q8_0"

	case MostlyQ5_0:
		return "MOSTLY_Q5_0"

	case MostlyQ5_1:
		return "MOSTLY_Q5_1"

	case MostlyQ2_K:
		return "MOSTLY_Q2_K"

	case MostlyQ3_KS:
		return "MOSTLY_Q3_KS"

	case MostlyQ3_KM:
		return "MOSTLY_Q3_KM"

	case MostlyQ3_KL:
		return "MOSTLY_Q3_KL"

	case MostlyQ4_KS:
		return "MOSTLY_Q4_KS"

	case MostlyQ4_KM:
		return "MOSTLY_Q4_KM"

	case MostlyQ5_KS:
		return "MOSTLY_Q5_KS"

	case MostlyQ5_KM:
		return "MOSTLY_Q5_KM"

	case MostlyQ6_K:
		return "MOSTLY_Q6_K"

	default:
		return "UNKNOWN"
	}
}
