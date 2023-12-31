# SmoothedPlanetaryHydrodynamics
Java implementation of smoothed particle hydrodynamics (SPH), to simulate the collisions of planetary scale bodies in realtime using CUDA (~32,000 particles at 60 FPS on an RTX 4090, 64,000 at 30 FPS).
![sim1](https://github.com/Jazz-Coding/SmoothedPlanetaryHydrodynamics/assets/52354702/270775ec-8975-4c66-baaf-db7d80226f05)
_A snapshot from a collision simulation, shortly following collision of a small mass with an Earth-sized mass._

![simulation_collage](https://github.com/Jazz-Coding/SmoothedPlanetaryHydrodynamics/assets/52354702/8ab87e1c-98db-4b40-9a7b-aac09b9af6eb)
_Moon formation simulation: planet Theia impacting the proto-Earth at a 45 degree angle at 8km/s._

SPH simulates matter in four steps:

1. Matter is assumed to have continuous fields like density, pressure, and velocity.
2. The continuum is divided into a finite collection of discrete elements, "particles".
3. These particles interact pairwise (n^2), each particle influencing their neighbors weighted by a special "smoothing kernel", which weights more distant interactions weakly.
4. The sum of these interactions provides a net force determined through the Euler-Lagrange equations. This is then integrated over time to compute new velocities and positions on the next timestep
   (a finite subdivision of time, which is also assumed to be continuous). This is repeated throughout the simulation.

Main.java contains the rendering pipeline and dispatches to SPH_GPU.
gpulib/SPH_GPU.java is the core of the SPH simulation.

Currently, to execute, you must clone the repository and run from source code (I am in the process of building a JAR release file). JDK version 19 was used for development.
Keybinds:
 - PAUSE BREAK = Pause time.
 - UP/DOWN Arrows = Accelerate/Deaccelerate time (may destabilize the simulation if increased too far).
 - BACKSPACE = Reverse time.
 - Right click = Orbit camera target.
 - Shift + Right click = Translate camera left/right/up/down relative to the camera.
 - Scroll = Zoom out
 - 0 = Zero all velocities.
