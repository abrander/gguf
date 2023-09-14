package gguf

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
