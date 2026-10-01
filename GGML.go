package gguf

import (
	"fmt"
)

// GGML is used to represent the encoding of tensor data.
type GGML int

const (
	GgmlF32        GGML = 0
	GgmlF16        GGML = 1
	GgmlQ4_0       GGML = 2
	GgmlQ4_1       GGML = 3
	GgmlQ4_2       GGML = 4 // support has been removed
	GgmlQ4_3       GGML = 5 // support has been removed
	GgmlQ5_0       GGML = 6
	GgmlQ5_1       GGML = 7
	GgmlQ8_0       GGML = 8
	GgmlQ8_1       GGML = 9
	GgmlQ2_K       GGML = 10
	GgmlQ3_K       GGML = 11
	GgmlQ4_K       GGML = 12
	GgmlQ5_K       GGML = 13
	GgmlQ6_K       GGML = 14
	GgmlQ8_K       GGML = 15
	GgmlIQ2_XXS    GGML = 16
	GgmlIQ2_XS     GGML = 17
	GgmlIQ3_XXS    GGML = 18
	GgmlIQ1_S      GGML = 19
	GgmlIQ4_NL     GGML = 20
	GgmlIQ3_S      GGML = 21
	GgmlIQ2_S      GGML = 22
	GgmlIQ4_XS     GGML = 23
	GgmlI8         GGML = 24
	GgmlI16        GGML = 25
	GgmlI32        GGML = 26
	GgmlI64        GGML = 27
	GgmlF64        GGML = 28
	GgmlIQ1_M      GGML = 29
	GgmlBF16       GGML = 30
	GgmlQ4_0_4_4   GGML = 31 // support has been removed from gguf files
	GgmlQ4_0_4_8   GGML = 32 // support has been removed from gguf files
	GgmlQ4_0_8_8   GGML = 33 // support has been removed from gguf files
	GgmlTQ1_0      GGML = 34
	GgmlTQ2_0      GGML = 35
	GgmlIQ4_NL_4_4 GGML = 36 // support has been removed..?
	GgmlIQ4_NL_4_8 GGML = 37 // support has been removed..?
	GgmlIQ4_NL_8_8 GGML = 38 // support has been removed..?
	GgmlMXFP4      GGML = 39 // MXFP4 (1 block)
	GgmlNVFP4      GGML = 40 // NVFP4 (4 blocks, E4M3 scale)
	GgmlQ1_0       GGML = 41
	GgmlQ2_0       GGML = 42
	GgmlCount      GGML = 43

	// Aliases for backwards-compatibility.
	GgmlFloat32 = GgmlF32
	GgmlFloat16 = GgmlF16
)

// String returns the string representation of the encoding.
// Implements fmt.Stringer.
func (g GGML) String() string {
	switch g {
	case GgmlF32:
		return "F32"
	case GgmlF16:
		return "F16"
	case GgmlQ4_0:
		return "Q4_0"
	case GgmlQ4_1:
		return "Q4_1"
	case GgmlQ4_2:
		return "Q4_2"
	case GgmlQ4_3:
		return "Q4_3"
	case GgmlQ5_0:
		return "Q5_0"
	case GgmlQ5_1:
		return "Q5_1"
	case GgmlQ8_0:
		return "Q8_0"
	case GgmlQ8_1:
		return "Q8_1"
	case GgmlQ2_K:
		return "Q2_K"
	case GgmlQ3_K:
		return "Q3_K"
	case GgmlQ4_K:
		return "Q4_K"
	case GgmlQ5_K:
		return "Q5_K"
	case GgmlQ6_K:
		return "Q6_K"
	case GgmlQ8_K:
		return "Q8_K"
	case GgmlIQ2_XXS:
		return "IQ2_XXS"
	case GgmlIQ2_XS:
		return "IQ2_XS"
	case GgmlIQ3_XXS:
		return "IQ3_XXS"
	case GgmlIQ1_S:
		return "IQ1_S"
	case GgmlIQ4_NL:
		return "IQ4_NL"
	case GgmlIQ3_S:
		return "IQ3_S"
	case GgmlIQ2_S:
		return "IQ2_S"
	case GgmlIQ4_XS:
		return "IQ4_XS"
	case GgmlI8:
		return "I8"
	case GgmlI16:
		return "I16"
	case GgmlI32:
		return "I32"
	case GgmlI64:
		return "I64"
	case GgmlF64:
		return "F64"
	case GgmlIQ1_M:
		return "IQ1_M"
	case GgmlBF16:
		return "BF16"
	case GgmlQ4_0_4_4:
		return "Q4_0_4_4"
	case GgmlQ4_0_4_8:
		return "Q4_0_4_8"
	case GgmlQ4_0_8_8:
		return "Q4_0_8_8"
	case GgmlTQ1_0:
		return "TQ1_0"
	case GgmlTQ2_0:
		return "TQ2_0"
	case GgmlIQ4_NL_4_4:
		return "IQ4_NL_4_4"
	case GgmlIQ4_NL_4_8:
		return "IQ4_NL_4_8"
	case GgmlIQ4_NL_8_8:
		return "IQ4_NL_8_8"
	case GgmlMXFP4:
		return "MXFP4"
	case GgmlNVFP4:
		return "NVFP4"
	case GgmlQ1_0:
		return "Q1_0"
	case GgmlQ2_0:
		return "Q2_0"
	case GgmlCount:
		return "Count"
	default:
		return fmt.Sprintf("GGML(%d)", g)
	}
}
