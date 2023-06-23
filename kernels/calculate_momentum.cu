extern "C"
__global__ void calculate_momentum(float* momenta, float* densities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N_particles) return;

    float rho = densities[idx];
    float P = K*pow(rho,adiabaticIndex); // Equation of state, in this case, the ideal gas equation.
    momenta[idx] = P / (rho * rho);
}