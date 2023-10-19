package gguf

const qK4_0 = 32
const qK4_1 = 32
const qK5_0 = 32
const qK5_1 = 32
const qK8_0 = 32
const qK_K = 256

const kScaleSize = 12

var sizes = map[GGML]struct {
	blocksize     uint64
	valuesinblock uint64
}{
	GgmlFloat32: {blocksize: 4, valuesinblock: 4},
	GgmlFloat16: {blocksize: 2, valuesinblock: 2},
	GgmlQ4_0:    {blocksize: 2 + qK4_0/2, valuesinblock: qK4_0},
	GgmlQ4_1:    {blocksize: 4 + qK4_1/2, valuesinblock: qK4_1},
	GgmlQ5_0:    {blocksize: 2 + 4 + qK5_0/2, valuesinblock: qK5_0},
	GgmlQ5_1:    {blocksize: 4 + 4 + qK5_1/2, valuesinblock: qK5_1},
	GgmlQ8_0:    {blocksize: 2 + qK8_0, valuesinblock: qK8_0},
	GgmlQ8_1:    {blocksize: 2 + 2 + qK8_0, valuesinblock: qK8_0},
	GgmlQ2_K:    {blocksize: qK_K/2 + qK_K/4 + 2 + 2, valuesinblock: qK_K},
	GgmlQ3_K:    {blocksize: qK_K/8 + qK_K/4 + 2 + 2, valuesinblock: qK_K},
	GgmlQ4_K:    {blocksize: 2*2 + 2 + qK_K/2, valuesinblock: qK_K},
	GgmlQ5_K:    {blocksize: 2 + 2 + kScaleSize + qK_K/2, valuesinblock: qK_K},
	GgmlQ6_K:    {blocksize: qK_K/2 + qK_K/4 + qK_K/16 + 2, valuesinblock: qK_K},
	GgmlQ8_K:    {blocksize: 4 + qK_K + 2*qK_K/16, valuesinblock: qK_K},
	GgmlInt8:    {blocksize: 1, valuesinblock: 1},
	GgmlInt16:   {blocksize: 2, valuesinblock: 2},
	GgmlInt32:   {blocksize: 4, valuesinblock: 4},
}
