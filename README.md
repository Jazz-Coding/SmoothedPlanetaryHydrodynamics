# SmoothedPlanetaryHydrodynamics
Java implementation of smoothed particle hydrodynamics (SPH), to simulate the collisions of planetary scale bodies.

SPH simulates matter in four steps:
1. Matter is assumed to have continuous fields like density, pressure, and velocity.
2. The continuum is divided into a finite collection of discrete elements, "particles".
3. These particles interact pairwise (n^2), each particle influencing their neighbors weighted by a special "smoothing kernel", which weights more distant interactions weakly.
4. The sum of these interactions provides a net force determined through the Euler-Lagrange equations. This is then integrated over time to compute new velocities and positions on the next timestep
   (a finite subdivision of time, which is also assumed to be continuous). This is repeated throughout the simulation.

Main.java contains the rendering pipeline and dispatches to SPH_GPU.
gpulib/SPH_GPU.java is the core of the SPH simulation.
