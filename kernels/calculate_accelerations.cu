// Vector divergence.
// u = (hv_ij . r_ij) / (|r_ij|^2 + e*h^2)
__device__ float divergence(float VR, float3 displacement, float smoothingLength){
    float numerator = smoothingLength*VR;
    float denominator = norm1(displacement) + (epsilon*(smoothingLength*smoothingLength)); // Extra small term to prevent singularities.
    return numerator / denominator;
}

/**
    Artificial viscosity component.

    Introduced to handle supersonic shocks and flows that would otherwise make the velocity field discontinuous when particles would
    overshoot one another, and thus compromise the accuracy of the simulation.

    Combines two approaches, one considering the bulk (compressive/contractive) and shear (parallel) viscosities, and one using the squared
    velocity divergence (local compression/contraction). The former is more equipped to handle extreme shocks, whereas the latter is more suited
    for weak shocks. Consequently, the stronger one falls off quadratically whereas the weak one falls off linearly (so the weaker one becomes the dominant
    one for weak shocks).
*/

__device__ __forceinline__ float viscosity_VN(float velocityDivergence){
    return beta * velocityDivergence * velocityDivergence; // Squared velocity divergence.
}

__device__ __forceinline__ float viscosity_BS(float velocityDivergence, float speedOfSound){
    return -alpha * velocityDivergence * speedOfSound; // Bulk and shear viscosities based on speed of sound.
}

__device__ __forceinline__ float soundSpeed(float localDensity){
    return sqrt(adiabaticIndex * K * pow(localDensity,adiabaticIndex-1)); // sqrt(dP/dp)
}

__device__ float viscosity(float3 r_ij,
                           float3 v_ij, float smoothingLength,
                           float avgDensity, bool reversed){
    float VR = dot(v_ij, r_ij);
    if(reversed){
        if(VR <= 0){
            return 0; // Only consider convergent flows.
        }
    } else{
        if(VR >= 0){
          return 0;
        }
    }

    float u = divergence(VR, r_ij,smoothingLength);
    float v_bs = viscosity_BS(u,soundSpeed(avgDensity));
    float v_vn = viscosity_VN(u);

    return (v_bs + v_vn) / avgDensity;
}

/**
    The heart of the SPH simulation. Calculates the combined momentum of the particles, and thus their velocity.
*/
extern "C"
__global__ void calculate_accelerations(
                float* densities,
                float* velocities_raw,
                float* accelerations_raw, // Particle data
                float* momenta,
                float* positions_raw,
                float splineGradConstant, float gravGradConstant, float smoothingLength, bool reversed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_particles) return;

    // Cast pointers to float3s which are more concise and easier to work with.
    float3* positions = (float3*) positions_raw;
    float3* velocities = (float3*) velocities_raw;
    float3* accelerations = (float3*) accelerations_raw;

    // Change in acceleration of particle i due to pressure and gravity.
    float3 pos_i = positions[idx];

    // Properties for calculating fluid pressure.
    float momentum_i = momenta[idx];

    // Properties for calculating the viscosity term.
    float density_i = densities[idx];
    float3 velocity_i = velocities[idx];

    // Accumulators. Separated so that separate scalars can be applied at the end instead of per-particle, reducing the computational cost.
    float3 pressurePush = {0,0,0};
    float3 gravityPull = {0,0,0};

    float cutoffDistance = 2*smoothingLength;
    // Sum interactions with every other particle.
    for(int j = 0; j < N_particles; j++){
        if(j == idx) continue; // Avoid self-interaction.

        // Properties for calculating fluid pressure.
        float density_j = densities[j];
        float momentum_j = momenta[j];

        float3 pos_j = positions[j];
        float3 velocity_j = velocities[j];

        float3 displacement = sub(pos_i,pos_j); // r_ij = r_i-r_j
        float distance = norm2(displacement); // r = |r_ij|
        float3 unitVector = unit(displacement,distance);

        // Second component (gravity, no corrective factor applied).
        float F_g = gravGradSoften(distance, gravGradConstant, smoothingLength);
        addIP(gravityPull, scal(unitVector,F_g));

        if(distance > cutoffDistance) continue;


        // First component of the Lagrangian equations of motion.
        float W = cubicSplineGradient(distance, splineGradConstant, smoothingLength);
        float density_avg = (density_i+density_j) / 2;
        float3 v_ij = sub(velocity_i,velocity_j);
        float visc = viscosity(displacement, v_ij, density_avg, smoothingLength,reversed);
        float F_p = (momentum_i+momentum_j+visc)*W; // Smooth the interaction.

        addIP(pressurePush, scal(unitVector,F_p));
    }

    // Aggregate the net acceleration changes into global memory.
    float gravityScalar  = -particleMass*G/2;
    float pressureScalar = -particleMass;
    float3 netMotion = addScaled(gravityPull, pressurePush, gravityScalar, pressureScalar);
    setVector(accelerations[idx],netMotion);
}