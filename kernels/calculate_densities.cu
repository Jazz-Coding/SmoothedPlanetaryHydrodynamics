extern "C"
__global__ void calculate_densities(
    float* densities,
    float* positions_raw, float splineConst, float smoothingLength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_particles) return;

    // Cast pointers to float3s which are more concise and easier to work with.
    float3* positions = (float3*) positions_raw;
    float3 pos_i = positions[i];
    float cutoffDistance = 2*smoothingLength;

    float density_net = cubicSpline(0,splineConst,smoothingLength);

    for (int j = 0; j < N_particles; j++) {
        if(j == i) continue;

        float3 pos_j = positions[j];
        float3 displacement = sub(pos_i,pos_j);

        // Calculate the euclidean distance between particles i and j.
        float distance = norm2(displacement);
        if(distance>cutoffDistance) continue;

        // Calculate the density gradient between them as per the cubic spline function.
        float gradient = cubicSpline(distance,splineConst,smoothingLength);

        // Update the density array.
        density_net += gradient;
    }

    densities[i] = particleMass * density_net;
}