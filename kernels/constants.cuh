// Simulation properties
__constant__ int dimensions;
__constant__ int N_particles;

// Particle mass is assumed to be constant.
__constant__ float particleMass;
// Big G, universal gravitational constant.
__constant__ float G;

// Equation of state parameters.
__constant__ float K;
__constant__ float adiabaticIndex;

// Viscosity control parameters.
__constant__ float alpha;
__constant__ float beta;
__constant__ float epsilon;