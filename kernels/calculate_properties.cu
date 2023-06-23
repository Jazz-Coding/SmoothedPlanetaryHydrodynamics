extern "C"
__global__ void calculate_properties(
                float* densities, // Scalar property
                float* positions_raw, // Vector properties
                float* velocities_raw,
                float* accelerations_raw,
                float* momentumCalculated,
                float* kineticEnergyCalculated) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_particles) return;

    // Cast pointers to float3s which are more concise and easier to work with.
    float3* positions = (float3*) positions_raw;
    float3* velocities = (float3*) velocities_raw;
    float3* accelerations = (float3*) accelerations_raw;

    float density = densities[idx];
    float3 position = positions[idx];

    float3 velocity = velocities[idx];
    float speed = norm2(velocity);

    float3 acceleration = accelerations[idx];

    // Record evaluation data.
    momentumCalculated[idx] = particleMass * speed;
    kineticEnergyCalculated[idx] = 0.5 * particleMass * (speed*speed);
}