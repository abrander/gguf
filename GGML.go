package gguf

import (
	"fmt"
)

type GGML int

const (
	GgmlFloat32 GGML = 0
	GgmlFloat16 GGML = 1
	GgmlQ4_0    GGML = 2
	GgmlQ4_1    GGML = 3
	GgmlQ5_0    GGML = 6
	GgmlQ5_1    GGML = 7
	GgmlQ8_0    GGML = 8
	GgmlQ8_1    GGML = 9
	GgmlQ2_K    GGML = 10
	GgmlQ3_K    GGML = 11
	GgmlQ4_K    GGML = 12
	GgmlQ5_K    GGML = 13
	GgmlQ6_K    GGML = 14
	GgmlQ8_K    GGML = 15
	GgmlInt8    GGML = 16
	GgmlInt16   GGML = 17
	GgmlInt32   GGML = 18
)

func (g GGML) String() string {
	switch g {
	case GgmlFloat32:
		return "float32"
	case GgmlFloat16:
		return "float16"
	case GgmlQ4_0:
		return "q4_0"
	case GgmlQ4_1:
		return "q4_1"
	case GgmlQ5_0:
		return "q5_0"
	case GgmlQ5_1:
		return "q5_1"
	case GgmlQ8_0:
		return "q8_0"
	case GgmlQ8_1:
		return "q8_1"
	case GgmlQ2_K:
		return "q2_k"
	case GgmlQ3_K:
		return "q3_k"
	case GgmlQ4_K:
		return "q4_k"
	case GgmlQ5_K:
		return "q5_k"
	case GgmlQ6_K:
		return "q6_k"
	case GgmlQ8_K:
		return "q8_k"
	case GgmlInt8:
		return "int8"
	case GgmlInt16:
		return "int16"
	case GgmlInt32:
		return "int32"
	default:
		return fmt.Sprintf("GGML(%d)", g)
	}
}
