package com.jazz.gpulib;

import com.jazz.graphics.opengl.utils.Log;
import com.jazz.graphics.opengl.utils.LogLevel;
import com.jazz.math.VMath2;
import com.jazz.utils.Stopwatch;
import jcuda.driver.CUgraphicsResource;

import static com.jazz.gpulib.MatrixMath.*;
import static com.jazz.utils.Constants.*;
import static jcuda.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE;
import static jcuda.driver.JCudaDriver.*;

/**
 * The heart of the SPH simulation on the GPU.
 * Handles and orchestrates the many steps in the computation.
 */
public class SPH_GPU {
    private static double k_base = 1.380649e-23; // Boltzmann constant, presently only used for debugging purposes.
    private static double G_base = 6.6743015e-11; // Universal gravitational constant, G.

    private static Log log = new Log(LogLevel.DEBUG);

    /**
     * Converts the value G into the scene units.
     * This enables us to use smaller numbers such as "1" for the mass of the Earth.
     * To understand this, it is helpful to realize that the kilogram is itself an arbitrary unit,
     * and in fact we could have chosen any reference value, here, we choose the Earth.
     */
    private static double convertG(double metresPerLengthUnit,
                                   double kgPerMassUnit,
                                   double secondsPerTimeUnit){
        // G = 6.6743015e-11 * L^3 * M^-1 * T^-2
        // (L*2)^3 = 2^3*L^3, so we'd need to divide by 2^3 to "undo" this change.
        // (2*M)^-1 = 1/2 * 1/M, so we'd need to multiply by 2 to "undo" this change.
        double G_new = G_base;
        G_new *= Math.pow(secondsPerTimeUnit,2);
        G_new *= Math.pow(kgPerMassUnit,1);
        G_new /= Math.pow(metresPerLengthUnit,3);
        return G_new;
    }

    /**
     * The same process for the boltzmann constant. Presently only used for debugging.
     */
    private static double convertBoltzmannConstant( double metresPerLengthUnit,
                                                   double kgPerMassUnit,
                                                   double secondsPerTimeUnit,
                                                   double kelvinPerTempUnit){
        // k_b = 1.380649e-23 L^2 * M * T^-2 * TEMP^-1
        double k_new = k_base;
        k_new *= Math.pow(kelvinPerTempUnit,1);
        k_new *= Math.pow(secondsPerTimeUnit,2);
        k_new /= Math.pow(kgPerMassUnit,1);
        k_new /= Math.pow(metresPerLengthUnit,2);
        return k_new;
    }


    // Simulation units, Earth masses and radii.
    private static float UNIT_RADIUS = 1F;
    private static float UNIT_MASS = 1F * (0.887F+0.133F); // Initialized to the Earth-Theia collision.

    public static float G_ADJUSTED = (float) convertG(EARTH_RADIUS, EARTH_MASS, 1);

    // The boltzmann constant, experimental way of computing the polytropic constant for stars.
    public static double K_ADJUSTED = convertBoltzmannConstant(EARTH_RADIUS,1,1,1);
    private static double computePolytropicConstant(float starTemperature, float particleMass, float polytropicIndex){
        return ((k_base * starTemperature) / Math.pow(particleMass,polytropicIndex-1));
    }

    // Simulation parameters.
    public static float SMOOTHING_LENGTH = 1F;
    public static float SOFTENING_LENGTH = SMOOTHING_LENGTH;
    //public static float SMOOTHING_LENGTH_COUPLING_CONSTANT = 1.3F;

    // Equation of state parameters. The ideal gas law.
    public static float K;
    public static float adiabaticIndex;

    // Artificial viscosity parameters.
    public static float epsilon;
    public static float alpha;
    public static float beta;

    public static int N_PARTICLES;
    public static final int DIMENSIONS = 3;

    // Derived simulation units.
    public static float PARTICLE_MASS;


    // 2D debugging GUI.
    //private static BasicPanelGPU graphicsPanel;

    // Particle representation: matrices of physical properties like position and velocity.
    private static MatrixPointer _positions;
    private static MatrixPointer _velocities;
    private static MatrixPointer _accelerations;

    // Scalar data, stored as single vectors on the GPU.
    private static MatrixPointer _densities;
    private static MatrixPointer _momenta;

    // CPU copy for statistical analysis.
    public static float[][] positions;
    public static float[][] velocities;
    public static float[][] accelerations;
    public static float[] densities;

    // Statistical values:
    // Extreme values.
    public static float greatestParticleExtent=0;
    public static float highestVelocity=0;
    public static float highestAcceleration=0;
    public static float highestForce=0;
    public static float highestDensity=0;

    // Average values.
    public static float averageParticleExtent=0;
    public static float averageVelocity=0;
    public static float averageAcceleration=0;
    public static float averageForce=0;
    public static float averageDensity=0;

    /**
     *  Simulation initialization method
     *  - Multiple bodies distributed with different positions, masses, and initial velocities (linear).
     */
    private static void multiDistribute(float[] massRatios, float[][] bodyInitialPositions, float[][] bodyInitialVelocities,
                                        float[][] positionsArray, float[][] velocitiesArray){
        int bodies = massRatios.length;
        int[] particleCounts = new int[bodies];

        for (int i = 0; i < bodies; i++) {
            particleCounts[i] = (int) (massRatios[i] * N_PARTICLES);
            System.out.println(particleCounts[i]);
        }

        // Check we didn't miss any.
        int used = 0;
        for (int i = 0; i < bodies; i++) {
            used += particleCounts[i];
        }
        if(used < N_PARTICLES){
            int difference = N_PARTICLES-used;
            particleCounts[bodies-1]+=difference; // Add them to the last one.
        }

        // Now actually distribute the particles.
        int superindex=0;
        for (int i = 0; i < bodies; i++) {
            int particlesForThisBody = particleCounts[i];
            float[] positionOffset = bodyInitialPositions[i];
            float[] velocity = bodyInitialVelocities[i];

            float effectiveRadius = (float) Math.cbrt(UNIT_MASS*massRatios[i]);
            for (int j = superindex; j < superindex+particlesForThisBody; j++) {
                positionsArray[j] = VMath2.randomNSphere(effectiveRadius, DIMENSIONS);
                VMath2.vsubIP(positionsArray[j], positionOffset);
                VMath2.vsubIP(velocitiesArray[j],velocity);
            }
            superindex+=particlesForThisBody;
        }
    }

    /**
     * Simulation initialization method
     * - Two bodies on a collision course.
     */
    private static void twoDistribute(float[][] positions, float distributionRadius, float[][] emptyVelocities){
        // A simple 0.5M <------- 0.5M collision.
        float shareOfParticles = 0.6F;
        float otherShareOfParticles = 1F-shareOfParticles;
        for (int i = 0; i < N_PARTICLES*shareOfParticles; i++) {
            positions[i] = VMath2.randomNSphere(UNIT_RADIUS, DIMENSIONS);
            //positions[i][0]=0; //-= distributionRadius; // Offset by some distance on the x axis.
            //emptyVelocities[i] = new float[]{0, 0, 0};
        }

        for (int i = (int) (N_PARTICLES*shareOfParticles); i < N_PARTICLES; i++) {
            positions[i] = VMath2.randomNSphere(UNIT_RADIUS, DIMENSIONS);
            positions[i][0] += distributionRadius; // Offset by some distance on the x axis.
            positions[i][1] += 1;
            emptyVelocities[i] = new float[]{-0.001F, 0F, 0};
        }

        /*for (int i = (int) (N_PARTICLES*(shareOfParticles+otherShareOfParticles/2)); i < N_PARTICLES; i++) {
            positions[i] = VMath2.randomNSphere(UNIT_RADIUS/4, DIMENSIONS);
            positions[i][0] += distributionRadius/2; // Offset by some distance on the x axis.
            positions[i][1] += 0.5;
            emptyVelocities[i] = new float[]{0.0006F, 0F, 0};
        }*/
    }

    // Halts all particles, allows messing with the simulation via user input.
    public static void zeroVelocities(){
        MatrixMath.executeScale(_velocities,0);
    }

    /**
     * Simulation initialization method (EXPERIMENTAL)
     * - Uniformly distributed particles, tends to give more stable results as forces are in equilibrium
     * from the beginning.
     */
    private static void gridDistribute(float[][] positions, double radius){
        int n = (int) Math.cbrt(N_PARTICLES);
        int half = (n/2);
        double centre = 0;

        float downscale = 0.03f;

        int superindex = 0;
        for (int x = -half; x < half; x++) {
            for (int y = -half; y < half; y++) {
                for (int z = -half; z < half; z++) {
                    double dx = x-centre;
                    double dy = y-centre;
                    double dz = z-centre;
                    double distance = Math.sqrt(dx*dx+dy*dy+dz*dz);

                    if (distance <= radius*0.99) {
                        positions[superindex] = new float[]{x*downscale,y*downscale,z*downscale};
                        superindex++;
                    }
                }
            }
        }

        System.out.println(superindex);
        N_PARTICLES = superindex;
        setSimulationParameters(N_PARTICLES);
        //System.exit(0);
    }

    // Precomputed factors in the SPH equations, saves recomputing for each particle.
    public static float densitySmoothingConstant;
    public static float densitySmoothingGradientConstant;
    public static float gravitationalSofteningGradientConstant;

    private static float inversePiPower(double x, double power){
        double powered = Math.pow(x,power);
        double pied = Math.PI*powered;
        return (float) (1/pied);
    }
    private static float inversePower(double x, double power){
        double powered = Math.pow(x,power);
        return (float) (1/powered);
    }

    public static void updateSmoothingLength(float newValue){
        SMOOTHING_LENGTH = newValue;
        SOFTENING_LENGTH = SMOOTHING_LENGTH;

        // Update kernel constants.
        densitySmoothingConstant = inversePiPower(SMOOTHING_LENGTH,3); // 1/pi*h^3
        densitySmoothingGradientConstant = inversePiPower(SMOOTHING_LENGTH,4); // 1/4*pi*h^4
        gravitationalSofteningGradientConstant = inversePower(SMOOTHING_LENGTH,2); // 1/h^2
    }

    /**
     * Simulation initialization.
     * Here the initial conditions are described - for instance if there are many bodies or just one,
     * their initial velocities, their angle of impact (if any), etc.
     */
    private static void initializeParticlesOnGPU(){
        float[][] emptyVelocities = new float[N_PARTICLES][DIMENSIONS];
        float[][] emptyPositions = new float[N_PARTICLES][DIMENSIONS];

        float massTotal = 0.887F+0.133F;

        float radius = 0.887F;
        float impactAngle = (float) Math.toRadians(45);
        float startDistance = 15;
        float startVel = 0.00141265107F; // 11.2 km/s = 1* escape velocity
        float h = (float) (radius * Math.sin(impactAngle));

        multiDistribute(
                new float[]{0.887F/massTotal,0.133F/massTotal},          // Mass distribution.
                new float[][]{{0,0,0},{-startDistance,h,0}}, // Initial positions.
                new float[][]{{0,0,0},{startVel,0,0}}, // Initial velocities.
                emptyPositions,emptyVelocities);

        _positions = new MatrixPointer(emptyPositions,true);
        _velocities = new MatrixPointer(emptyVelocities,true);
        _accelerations = new MatrixPointer(new float[N_PARTICLES][DIMENSIONS],true);

        _densities = new MatrixPointer(new float[N_PARTICLES]);
        _momenta = new MatrixPointer(new float[N_PARTICLES]);
    }


    public static CUgraphicsResource vbo_resource;
    public static void prepCudaInterop(int vbo){
        vbo_resource = new CUgraphicsResource();

        // Register the VBO with CUDA
        cuGraphicsGLRegisterBuffer(vbo_resource, vbo, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    }

    public static void GPU_GUI_UPDATE(){
        int offset = 80*2*3; // 80 vertices per mesh, 2 properties before the offset in the same VBO (position and color), 3 floating point values each (XYZ).
        MatrixMath.GPU_GUI_UPDATE(_positions,vbo_resource,N_PARTICLES,offset);
    }

    private static int t;
    private static Stopwatch stopwatch;

    public static void init(int particles){
        setSimulationParameters(particles);

        // Initialize GPU engine.
        MatrixMath.initialize();
        initializeParticlesOnGPU();

        updateSmoothingLength(SMOOTHING_LENGTH);
        setGPUConstants(DIMENSIONS,N_PARTICLES,PARTICLE_MASS, G_ADJUSTED,K,adiabaticIndex,epsilon,alpha, beta);
        stopwatch = new Stopwatch();
    }

    private static float average(float[] values){
        float sum = 0f;
        for (float value : values) {
            sum+=value;
        }
        return sum/values.length;
    }

    public static void extrapolate(MatrixPointer velocities_half, MatrixPointer positions_half, MatrixPointer accelerations_half, boolean reversed){
        /**
         * Use current particle positions to determine local density.
         * Time complexity O(n^2)
         */
        stopwatch.start();
        updateDensitiesGPU(_densities, positions_half, densitySmoothingConstant,SMOOTHING_LENGTH);
        stopwatch.stop();
        log.print("Update densities time: " + stopwatch.durationMS() + " ms",LogLevel.PERFORMANCE);

        /**
         * Link density and pressure to fluid momentum as per the Euler-Lagrange equations.
         * Performing this step here enables it to be done in O(n) time complexity, since it would otherwise
         * need to be recomputed n times since parallel threads cannot share data efficiently.
         * Time complexity O(n).
         */
        stopwatch.start();
        updateMomentumGPU(_momenta,_densities);
        stopwatch.stop();
        log.print("Update momenta time: " + stopwatch.durationMS() + " ms",LogLevel.PERFORMANCE);

        /**
         * Combine fluid momentum, viscosity, and gravitational potential to determine the net momentum, and thus the acceleration,
         * for each particle.
         * Time complexity O(n^2)
         */
        stopwatch.start();
        updateAccelerationsGPU(_densities, velocities_half, accelerations_half, _momenta, positions_half, densitySmoothingGradientConstant,gravitationalSofteningGradientConstant,SOFTENING_LENGTH,reversed);
        stopwatch.stop();
        log.print("Update accelerations time: " + stopwatch.durationMS() + " ms",LogLevel.PERFORMANCE);
    }


    public static void doTimestep(float timestep){
        // Dynamic updates.
        //log.print("-----------------\nt="+t,LogLevel.DEBUG);
        System.out.print(t + " ");
        long start = System.nanoTime();
        integration(timestep,_positions,_velocities,_accelerations);
        long end = System.nanoTime();
        long duration = end-start;
        float runtimeMS = duration/1e6F;
        log.print("Runtime: " + runtimeMS + " ms",LogLevel.PERFORMANCE);
        log.print("Effective framerate: " + (1000/runtimeMS) + " FPS",LogLevel.PERFORMANCE);
        t++;

        MatrixMath.computeAndLogEvalStats(PARTICLE_MASS, _densities,_positions,_velocities,_accelerations,N_PARTICLES);
    }

    public static void setSimulationParameters(int particles){
        N_PARTICLES = particles;
        PARTICLE_MASS = UNIT_MASS/N_PARTICLES;

        // Equation of state.
        K = 0.000001F;
        adiabaticIndex = 8/3F;

        // Viscosity parameters.
        epsilon = 0.1F; // Anti-singularity term
        alpha = 1F; // Weak shock viscosity (linear fall off)
        beta = 2F; // Strong shock viscosity (quadratic fall off)

        SMOOTHING_LENGTH = 0.1F;

        updateSmoothingLength(SMOOTHING_LENGTH);
    }
}
