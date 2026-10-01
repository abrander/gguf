package gguf

const qK1_0 = 128
const qK2_0 = 64
const qK4_0 = 32
const qK4_1 = 32
const qK4_NL = 32
const qK5_0 = 32
const qK5_1 = 32
const qK8_0 = 32
const qK_K = 256
const qK_MXFP4 = 32
const qK_NVFP4 = 64

const kScaleSize = 12

// sizes is a map of GGML to the blocksize and number of values in a
// block. It's used to calculate the size of a tensor in Size().
var sizes = map[GGML]struct {
	blocksize     uint64 // Size in bytes
	valuesinblock uint64 // Number of values in the block
}{
	GgmlF32:     {blocksize: 4, valuesinblock: 1},
	GgmlF16:     {blocksize: 2, valuesinblock: 1},
	GgmlQ4_0:    {blocksize: 2 + qK4_0/2, valuesinblock: qK4_0},
	GgmlQ4_1:    {blocksize: 4 + qK4_1/2, valuesinblock: qK4_1},
	GgmlQ5_0:    {blocksize: 2 + 4 + qK5_0/2, valuesinblock: qK5_0},
	GgmlQ5_1:    {blocksize: 4 + 4 + qK5_1/2, valuesinblock: qK5_1},
	GgmlQ8_0:    {blocksize: 2 + qK8_0, valuesinblock: qK8_0},
	GgmlQ8_1:    {blocksize: 2 + 2 + qK8_0, valuesinblock: qK8_0},
	GgmlQ1_0:    {blocksize: 2 + qK1_0/8, valuesinblock: qK1_0},
	GgmlQ2_0:    {blocksize: 2 + qK2_0/4, valuesinblock: qK2_0},
	GgmlIQ2_XXS: {blocksize: 2 + qK_K/8*2, valuesinblock: qK_K},
	GgmlIQ2_XS:  {blocksize: 2 + qK_K/8*2 + qK_K/32, valuesinblock: qK_K},
	GgmlIQ3_XXS: {blocksize: 2 + 3*(qK_K/8), valuesinblock: qK_K},
	GgmlIQ1_S:   {blocksize: 2 + qK_K/8 + qK_K/16, valuesinblock: qK_K},
	GgmlIQ4_NL:  {blocksize: 2 + qK4_NL/2, valuesinblock: qK4_NL},
	GgmlIQ3_S:   {blocksize: 2 + 13*(qK_K/32) + qK_K/64, valuesinblock: qK_K},
	GgmlIQ2_S:   {blocksize: 2 + qK_K/4 + qK_K/16, valuesinblock: qK_K},
	GgmlIQ4_XS:  {blocksize: 2 + 2 + qK_K/64 + qK_K/2, valuesinblock: qK_K},
	GgmlI8:      {blocksize: 1, valuesinblock: 1},
	GgmlI16:     {blocksize: 2, valuesinblock: 1},
	GgmlI32:     {blocksize: 4, valuesinblock: 1},
	GgmlI64:     {blocksize: 8, valuesinblock: 1},
	GgmlF64:     {blocksize: 8, valuesinblock: 1},
	GgmlIQ1_M:   {blocksize: qK_K/8 + qK_K/16 + qK_K/32, valuesinblock: qK_K},
	GgmlBF16:    {blocksize: 2, valuesinblock: 1},
	GgmlTQ1_0:   {blocksize: 2 + qK_K/64 + (qK_K-4*qK_K/64)/5, valuesinblock: qK_K},
	GgmlTQ2_0:   {blocksize: 2 + qK_K/4, valuesinblock: qK_K},
	GgmlMXFP4:   {blocksize: 1 + qK_MXFP4/2, valuesinblock: qK_MXFP4},
	GgmlNVFP4:   {blocksize: 4 + qK_NVFP4/2, valuesinblock: qK_NVFP4},
	GgmlQ2_K:    {blocksize: 2*2 + qK_K/16 + qK_K/4, valuesinblock: qK_K},
	GgmlQ3_K:    {blocksize: 2 + qK_K/4 + qK_K/8 + kScaleSize, valuesinblock: qK_K},
	GgmlQ4_K:    {blocksize: 2*2 + kScaleSize + qK_K/2, valuesinblock: qK_K},
	GgmlQ5_K:    {blocksize: 2*2 + kScaleSize + qK_K/2 + qK_K/8, valuesinblock: qK_K},
	GgmlQ6_K:    {blocksize: qK_K/2 + qK_K/4 + qK_K/16 + 2, valuesinblock: qK_K},
	GgmlQ8_K:    {blocksize: 4 + qK_K + 2*qK_K/16, valuesinblock: qK_K},
}
