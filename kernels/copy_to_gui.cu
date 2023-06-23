#include "constants.cuh"

extern "C"
__global__ void copy_to_gui(
                float* positions_raw, float* offsets_raw, int vbo_offset_pos, int N_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_particles) return;

    float3* positions = (float3*) positions_raw;
    float4* offsets = (float4*) offsets_raw;

    float3 pos_i = positions[idx];

    // Update the positions.
    float scaleFactor = 5;
    offsets[vbo_offset_pos+idx] = {pos_i.x*scaleFactor, pos_i.y*scaleFactor, pos_i.z*scaleFactor, 1};
}