package com.jazz;

import com.jazz.graphics.BasicPanel;
import com.jazz.math.VMath2;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.jazz.SmoothingFunctions.*;

/**
 * CPU implementation with 2D graphics for debugging purposes.
 * Unused.
 */
public class SPH_CPU {
    /**
     * Particle properties.
     */
    private static float[] masses;
    private static float[] volumes;
    private static float[][] positions;
    private static float[][] velocities;
    private static float[][] accelerations;
    private static float[] densities;
    private static float[] pressures;

    /**
     * Simulation and physical properties.
     */
    private static int maxTimeSteps;
    private static float timeStep;

    private static int dimensions;
    private static float smoothingLength;

    private static int N_particles;
    private static float particleRadius;
    private static float particleMass = 0f;

    // Fluid properties
    private static float pressureConstant;
    private static float polytropicIndex;
    private static float damp;


    // Useful constants.
    private static final float PI = (float) Math.PI; // 32-bit version for higher performance/fewer downcasts.
    private static final float G = (float) 6.6743E-11;
    private static final float earthRadius = 6_371_000;

    // Multi-threaded values.
    private static int threads;
    private static int threadShare;

    // Particles each thread should handle, i.e. chunk size
    private static List<Callable<Object>> accelerationSimulators = new ArrayList<>();
    private static List<Callable<Object>> densitySimulators = new ArrayList<>();


    /**
     * Updates density values for all particles in the simulation.
     */

    public static void calculateDensities(ExecutorService es) throws InterruptedException {
        // Set up constant values.
        splineConstRegular = SmoothingFunctions.splineConstants[1];
        densitySelfContribution = particleMass * cubicSpline(0,
                                                            smoothingLength,
                                                            splineConstRegular);

        // Dispatch the simulators.
        es.invokeAll(densitySimulators);
    }


    private static float mu(float epsilon, int i, int j){
        float[] distance = diff(positions[i], positions[j]);
        float h2 = smoothingLength * smoothingLength;

        float[] deltaV = diff(velocities[i], velocities[j]);
        float dp = dot(deltaV,distance);
        return (smoothingLength*dp) / (norm1(distance) + epsilon*h2);
    }

    private static float divergence(float[] path, float[] velocityA, float[] velocityB){
        float[] deltaV = diff(velocityA, velocityB);
        float divergence = dot(deltaV,path) / (norm1(path));
        return divergence;
    }

    private static float viscosity(float alpha, float beta, float mu,
                                   int i, int j){
        //float u = (smoothingLength * dp) / ((distNorm * distNorm) * (epsilon * smoothingLength * smoothingLength));
        float speedOfSound = 330f;
        float avgDensity = ((densities[i] + densities[j])/2);

        float mu2 = mu*mu;

        return ((-alpha * speedOfSound * mu) +
                (beta * mu2)) /
                avgDensity;
    }

    private static boolean flowConvergent(int i, int j){
        /*float deltaP_x = positions[i][0] - positions[j][0];
        float deltaP_y = positions[i][1] - positions[j][1];

        float deltaV_x = velocities[i][0] - velocities[j][0];
        float deltaV_y = velocities[i][1] - velocities[j][1];*/

        float[] deltaV = diff(velocities[i], velocities[j]);
        float[] deltaP = diff(positions[i], positions[j]);
        float dp = dot(deltaV,deltaP);
        //float dotProduct = (deltaP_x*deltaV_x) + (deltaP_y*deltaV_y);

        return dp < 0;
    }

    private static void zeroAccelerations(){
        for (int i = 0; i < accelerations.length; i++) {
            accelerations[i][0] = 0f;
            accelerations[i][1] = 0f;
            //accelerations[i][2] = 0f;
        }
    }
    // Calculate acceleration due to gravity.
    public static void updateAccelerations(ExecutorService es) throws InterruptedException {
        // Reset the accelerations.
        zeroAccelerations();

        // Calculate spline constant.
        splineConstGradient = splineConstants[dimensions - 1] / (float) Math.pow(smoothingLength, dimensions);

        // Dispatch the simulators.
        es.invokeAll(accelerationSimulators);
    }

    private static float permanentPressure;
    private static float calculateMomentum(int i){
        return pressures[i]/(densities[i]*densities[i]);
    }
    public static float dot(float[] vecA, float[] vecB){
        float sum = 0f;
        sum += vecA[0]*vecB[0];
        sum += vecA[1]*vecB[1];
        sum += vecA[2]*vecB[2];
        return sum;
    }
    public static void vaddACC(float[] acc, float[] vecA, float[] vecB){
        acc[0] = vecA[0] + vecB[0];
        acc[1] = vecA[1] + vecB[1];
        acc[2] = vecA[2] += vecB[2];
    }
    public static void vsubACC(float[] acc, float[] vecA, float[] vecB){
        acc[0] = vecA[0] - vecB[0];
        acc[1] = vecA[1] - vecB[1];
        acc[2] = vecA[2] += vecB[2];
    }
    public static void vaddIP(float[] vecA, float[] vecB){
        vecA[0] += vecB[0];
        vecA[1] += vecB[1];
        vecA[2] += vecB[2];
    }
    private static void vsubIP(float[] vecA, float[] vecB) {
        vecA[0] -= vecB[0];
        vecA[1] -= vecB[1];
        vecA[2] -= vecB[2];
    }
    private static float[] vmultiply(float[] vector, float scalar){
        float[] product = new float[2];
        product[0] = vector[0]*scalar;
        product[1] = vector[1]*scalar;
        product[2] = vector[2]*scalar;
        return product;
    }
    private static float[] diff(float[] vecA, float[] vecB) {
        float[] diff = new float[2];
        diff[0] = vecA[0] - vecB[0];
        diff[1] = vecA[1] - vecB[1];
        diff[2] = vecA[2] - vecB[2];
        return diff;
    }

    private static float[] diffAbs(float[] vecA, float[] vecB) {
        float[] diff = new float[2];
        diff[0] = Math.abs(vecA[0] - vecB[0]);
        diff[1] = Math.abs(vecA[1] - vecB[1]);
        //diff[2] = vecA[2] - vecB[2];
        return diff;
    }
    private static float norm(float[] vecA) {
        float norm =
                        vecA[0] * vecA[0] +
                        vecA[1] * vecA[1];//+
        //vecA[2] * vecA[2];
        return norm;
    }
    private static float norm1(float[] vecA) {
        float norm1 =
                        Math.abs(vecA[0]) +
                        Math.abs(vecA[1]);//+
        //vecA[2] * vecA[2];
        return norm1;
    }
    private static float norm2(float[] vecA) {
        float norm2 =
                        vecA[0] * vecA[0] +
                        vecA[1] * vecA[1];//+
                        //vecA[2] * vecA[2];
        return (float) Math.sqrt(norm2);
    }

    private static float speedOfSound = 3000;
    private static float speedOfSoundSquared = speedOfSound*speedOfSound;
    private static void calculateSolidPressures(){
        for (int i = 0; i < N_particles; i++) {
            float densityDelta = ((densities[i] - referenceDensity) / referenceDensity) * particleMass;
            if(densityDelta != 0 && !Float.isNaN(densityDelta) && Float.isFinite(densityDelta)) {
                pressures[i] = speedOfSoundSquared * densityDelta;
            }
        }
    }

    public static void calculatePressures(float specificHeatCapacity, float adiabaticIndex, float temperatureKelvin) {
        for (int i = 0; i < N_particles; i++) {
            // "If the calorically perfect gas approximation is used,
            // then the ideal gas law may also be expressed as follows..."
            pressures[i] = densities[i] *
                    (adiabaticIndex-1)*
                    (specificHeatCapacity * temperatureKelvin);
            // specific_heat(water) = 4182 J/kg.C
            // room_temperature = 295K
        }
    }



    // EXTREMELY crude rendering of the particles for debugging purposes
    private static float renderDisplayWidth;
    private static int renderedParticleCount = 0;

    public static void printSimulationStats(){
        float avgDensity = 0f;
        for (int i = 0; i < N_particles; i++) {
            avgDensity += densities[i];
        }
        avgDensity /= N_particles;
        System.out.println("Average density=" + avgDensity);
        System.out.println("Smoothing Length=" + smoothingLength + " m");
    }

    private static float[] earthMasses = new float[]{
            181428955.238F,
            3.2916466e+16F, // 2D
            5.972E24F //3D
    };

    private static float areaCircle(float radius){
        return PI*radius*radius;
    }
    private static float volumeSphere(float radius){
        return (radius*radius*radius)*PI*4/3;
    }

    private static float[] earthVolumes = new float[]{
            0,
            (float) (PI * Math.pow(earthRadius,2)), // 2D
            (float) ((4F/3) * PI * Math.pow(earthRadius,3)) //3D
    };


    private static float softeningLength;
    // Initialize a test simulation, random distribution, at rest, in a sphere.
    public static void initParameters(){
        N_particles = 5_000;
        dimensions = 2;

        velocityBefore = new float[N_particles][2];
        velocityAfter = new float[N_particles][2];

        // Simulation properties.
        maxTimeSteps = 100_000;

        // Fluid properties
        damp = 0.25F;


        float distributionRadius = 0.5f*earthRadius;
        renderDisplayWidth = distributionRadius/64;

        float earthVolume = earthVolumes[dimensions-1];

        particleMass = earthVolume/N_particles;
        particleRadius = earthRadius/N_particles;

        positions = new float[N_particles][dimensions];
        velocities = new float[N_particles][dimensions];
        accelerations = new float[N_particles][dimensions];

        masses = new float[N_particles];
        volumes = new float[N_particles];
        pressures = new float[N_particles];
        densities = new float[N_particles];
        momenta = new float[N_particles];

        // Collision distribution.
        //distributeParticles(distributionRadius);
        gridDistribute((int) particleRadius);

        // Particles masses and volumes, constant to begin with.
        /*for (int i = 0; i < N_particles; i++) {
            masses[i] = particleMass;
            volumes[i] = particleVolume;
        }*/

        graphicsPanel = new BasicPanel(
                positions, accelerations, velocities, densities, pressures,
                renderDisplayWidth, areaCircle(particleRadius));
    }

    private static void distributeParticles(float distributionRadius){
        // A simple 0.8M <------- 0.2M collision.
        float distribution = 0.8F;
        for (int i = 0; i < N_particles*distribution; i++) {
            vsubACC(positions[i],VMath2.randomNSphere(distributionRadius*distribution, dimensions), new float[]{1f * distributionRadius,0});
        }
        for (int i = (int) (N_particles*distribution); i < N_particles; i++) {
            vaddACC(positions[i], VMath2.randomNSphere(distributionRadius*(1-distribution), dimensions),new float[]{2f * distributionRadius,0});
        }
    }

    // Distributes the particles in a circular/spherical grid, used for solid-modelling.
    private static void gridDistribute(int spacingDistance){
        float trueRadius = (float) Math.sqrt((N_particles) / Math.PI);
        float v = trueRadius * (particleRadius + spacingDistance);
        int r = (int) v;

        int pIndex = 0;
        for (int x = -r; x < r+1; x+=(spacingDistance + particleRadius/2)) {
            int Y = (int) Math.sqrt(r*r - x*x);
            for (int y = -Y; y < Y+1; y+=(spacingDistance + + particleRadius/2)) {
                positions[pIndex] = new float[]{x,y};
                pIndex++;
            }
        }
    }

    private static void initGUI(){
        JFrame window = new JFrame("SPH Simulation");
        window.setContentPane(graphicsPanel);
        window.setSize(1024,1024);
        window.setLocationRelativeTo(null);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
    }

    private static float densitySelfContribution = 0f;
    private static float splineConstRegular = 0f;
    private static float splineConstGradient = 0f;

    private static float weightInteractionCubicSmoothing(float forceMagnitude, float distance){
        return particleMass * forceMagnitude * cubicSplineGradient(distance,smoothingLength,splineConstGradient);
    }

    private static float weightInteractionGravitationalSmoothing(float forceMagnitude, float distance){
        return (-G/2) * forceMagnitude * gravSoften(distance,smoothingLength);
    }

    private static void smoothInteractions(int particle){
        // Cache the relevant information for the current particle.
        float[] here = positions[particle];
        float densityHere = densities[particle];
        float pressureHere = pressures[particle];
        float momentumHere = pressureHere / (densityHere*densityHere);
        float[] velocityHere = velocities[particle];

        // Consider interactions with all the other particles.
        for (int otherParticle = particle+1; otherParticle < N_particles; otherParticle++) {
            float[] there = positions[otherParticle];

            float[] path = diff(there,here); // The directional path to this other particle.
            float distance = norm2(path); // The length of that path.

            float densityThere = densities[otherParticle];
            float pressureThere = pressures[otherParticle];
            float momentumThere = pressureThere / (densityThere*densityThere);
            float[] velocityThere = velocities[otherParticle];

            // Smooth the fluid pressure interaction.
            float pressureInteraction = weightInteractionCubicSmoothing(momentumHere + momentumThere, distance);

            // Incorporate the effects of viscosity.
            // First determine whether the fluid is contracting or expanding here.
            float particleDivergence = divergence(path, velocityHere, velocityThere);
            if(particleDivergence < 0){
                // Contraction.

            }

            // Smooth the gravitational interaction.
            float gravitationalInteraction = weightInteractionGravitationalSmoothing(particleMass, distance);
        }
    }

    private static float[] momenta;

    private static void calculateMomenta(){
        for (int i = 0; i < N_particles; i++) {
            // Applies the fluid momentum equation: momentum = P/ρ^2 = Pressure/(Density^2)
            momenta[i] = calculateMomentum(i)*particleMass;
        }
    }
    private static void updateAccelerationsMK2(){
        zeroAccelerations();

        // Pre-computables.
        splineConstGradient = splineConstants[dimensions - 1] / (float) Math.pow(smoothingLength, dimensions);
        calculateMomenta();
        float gravityConstantFactor = (G/2)*particleMass;

        for (int i = 0; i < N_particles; i++) {
            float[] netPull = new float[2];
            for (int j = i + 1; j < N_particles; j++) {
                float[] distance = diff(positions[j], positions[i]);
                float distNorm = norm2(distance);

                float gravGradient = gravityConstantFactor * gravGradSoften(distNorm, softeningLength);

                // Artificial viscosity.
                float viscosity = 0;

                float alpha = 1F;
                float beta = 1F;
                float epsilon = 0.01F;

                if(flowConvergent(i,j)){
                    viscosity = viscosity(alpha, beta, mu(epsilon, i, j), i, j);
                }

                float pressureGradient = (momenta[i]+momenta[j]+viscosity) * cubicSplineGradient(distNorm, smoothingLength, splineConstGradient);


                float[] gg = vmultiply(distance, gravGradient);

                float[] pg = vmultiply(distance, pressureGradient);

                // j's gravitational PULL on i
                vaddIP(netPull,gg);

                // j's pressure PUSH on i
                vsubIP(netPull,pg);

                // Sign flipped (add -> sub, sub -> add) for the reverse, symmetrical interactions.
                // i's symmetrical gravitational PULL on j
                vsubIP(accelerations[j], gg);

                // i's symmetrical pressure PUSH on j
                vaddIP(accelerations[j], pg);
            }
            vaddIP(accelerations[i], netPull);

            // Factor in damping as some negative acceleration proportional to the velocity.
            vSAIP(accelerations[i], velocities[i], -0.5F);
            //vsubIP(accelerations[i], damping);
        }
    }

    private static void initMultithreading(){
        threads = 32;
        //threadShare = 64;//(N_particles / threads) / 2;

        // Due to the symmetric optimizations, each particle in the outer loop ends up
        // doing fewer and fewer calculations until the final ones are only interacting with a couple of particles each.
        // This means a simple division of the outer loop into 32 segments is inefficient.
        // Instead, we will keep assigning more until the thread reaches a pre-calculated share of the work.
        int totalWork = (N_particles*N_particles)/2;
        threadShare = (totalWork / threads);

        int[] workAssignments = new int[threads];
        int[] workLoads = new int[threads];
        int lastIndex = 0;
        for (int thread = 0; thread < threads; thread++) {
            // Keep adding outer-loop iterations until this thread's work exceeds "threadShare".
            while (workLoads[thread] < threadShare){
                int nextWorkLoad = N_particles-lastIndex;
                if(nextWorkLoad == 0){
                    // No more work, this thread will have to make do with less work.
                    break;
                }
                workAssignments[thread]++;
                workLoads[thread] += nextWorkLoad;
                lastIndex++;
            }
        }

        System.out.println("WORK ASSIGNMENTS: ");
        System.out.println(Arrays.toString(workAssignments));

        System.out.println("WORK LOADS: ");
        System.out.println(Arrays.toString(workLoads));

        System.out.println("TOTAL WORK: " + totalWork);
        System.out.println("THREAD SHARE: " + threadShare);
        //System.exit(0);


        // Initialize the simulators.
        int beginIndex = 0;
        for (int thread = 0; thread < threads; thread++) {
            int assignment = workAssignments[thread];
            // This thread should handle beginIndex -> beginIndex+assignment outer iterations.
            int finalBeginIndex = beginIndex;
            accelerationSimulators.add(Executors.callable(() -> {
                for (int i = finalBeginIndex; i < Math.min(N_particles, finalBeginIndex + assignment); i++) {
                    //float[] damping = scal(velocities[i], damp);

                    //float ox = positions[i][0];
                    //float oy = positions[i][1];

                    float[] netPull = new float[2];
                    //float netPullX = 0f;
                    //float netPullY = 0f;

                    // P/ρ^2 -> Pressure/(Density^2)
                    //float pppA = calculateMomentum(i);
                    float momentum_I = pressures[i]/(densities[i]*densities[i]);

                    for (int j = i + 1; j < N_particles; j++) {
                        float[] distance = diff(positions[j], positions[i]);
                        float distNorm = norm2(distance);

                        float gravGradient =
                                -(G/2) *
                                particleMass*
                                gravGradSoften(distNorm, softeningLength);


                        // Artificial viscosity.
                        float viscosity = 0;

                        float alpha = 1F;
                        float beta = 25F;
                        float epsilon = 0.01f;

                        if(flowConvergent(j,i)){
                            viscosity =
                                    viscosity(alpha, beta, mu(epsilon, j, i), j, i);
                        }


                        float pressureGradient = (pressures[i]/(densities[i]*densities[i]) +
                                                 (pressures[j]/(densities[j]*densities[j])) - viscosity) *
                                                 cubicSplineGradient(distNorm, smoothingLength, splineConstGradient);
                        float[] gg = vmultiply(distance,gravGradient); // distance[] * scalar
                        //float gradX = distX*gravGradient;
                        //float gradY = distY*gravGradient;

                        float[] pg = vmultiply(distance, pressureGradient); // distance[] * scalar
                        //float pGradX = pressureGradient*distX;
                        //float pGradY = pressureGradient*distY;

                        // Flip the signs for j's influence on i since distance (the scalar multiplied by the gradients) is being measured from j's perspective, not i's.
                        vsubIP(netPull,gg); // just add it
                        //netPullX -= gradX;
                        //netPullY -= gradY;

                        vaddIP(netPull,pg);
                        //netPullX += pGradX;
                        //netPullY += pGradY;

                        // Pull on J
                        // Effects on J accounted for here.
                        vaddIP(accelerations[j], gg);
                        //accelerations[j][0] += gradX;
                        //accelerations[j][1] += gradY;

                        vaddIP(accelerations[j], pg);
                        //accelerations[j][0] -= pGradX;
                        //accelerations[j][1] -= pGradY;
                    }

                    // Effects on I accounted for here.
                    vaddIP(accelerations[i], netPull);
                    //accelerations[i][0] += netPullX;
                    //accelerations[i][1] += netPullY;

                    // Factor in damping as some negative acceleration proportional to the velocity.
                    vSAIP(accelerations[i], velocities[i], -damp);
                    //vsubIP(accelerations[i], damping);
                }
            }));
            System.out.println("This thread is doing particles " + finalBeginIndex + "-" + (finalBeginIndex+assignment));
            beginIndex += assignment;
        }

        // O(n) processes (like density calculation) can be divided more conventionally.
        int fixedThreadShare = N_particles/threads;
        for (int k = 0; k < N_particles; k+=fixedThreadShare) {
            int finalK = k;
            densitySimulators.add(Executors.callable(() -> {
                for (int i = finalK; i < Math.min(N_particles, finalK+fixedThreadShare); i++) {
                    densities[i] = densitySelfContribution;
                    for (int j = i+1; j < N_particles; j++) {
                        float normDist = norm2(diff(positions[j], positions[i]));
                        float gradient = particleMass * cubicSpline(normDist, smoothingLength, splineConstRegular);

                        densities[j] += gradient;
                        densities[i] += gradient;
                    }
                }
            }));
        }
    }

    private static BasicPanel graphicsPanel;
    private static float referenceDensity = 0f;

    // acc = scalar*vecA + vecB
    private static void vFMA(float[] acc, float[] vecA, float scalar, float[] vecB){
        acc[0] = scalar*vecA[0] + vecB[0];
        acc[1] = scalar*vecA[1] + vecB[1];
        //acc[2] = scalar*vecA[2] + vecB[2];
    }
    //vecA = vecA + scalar*vecB
    private static void vSAIP(float[] vecA, float[] vecB, float scalar){
        vecA[0] += scalar*vecB[0];
        vecA[1] += scalar*vecB[1];
        //vecA[2] += scalar*vecB[2];
    }

    private static void vINTACC(float[] acc, float[] vecA, float[] vecB){
        acc[0] = (vecA[0] + vecB[0])/2;
        acc[1] = (vecA[1] + vecB[1])/2;
        //acc[2] = (vecA[2] + vecB[2])/2;
    }

    private static void vScalAcc(float[] acc, float[] vec, float scalar){
        acc[0] += vec[0] * scalar;
        acc[1] += vec[1] * scalar;
        //acc[2] = vec[2] * scalar;
    }

    private static void vAssign(float[] vecA, float[] vecB){
        vecA[0] = vecB[0];
        vecA[1] = vecB[1];
        //vecA[2] = vecB[2];
    }

    /**
     * Leapfrog time-integration.
     */
    private static float[][] velocityBefore;
    private static float[][] velocityAfter;
    private static float velocityMax; // For use in dynamic time-stepping/smoothing length.
    private static void integrate(){
        velocityMax = 0;
        for (int i = 0; i < N_particles; i++) {
            // v+ = Δt*a + v-   // scaled addition (scalar = timestep)
            vFMA(velocityAfter[i], accelerations[i], timeStep, velocityBefore[i]);
            // x += Δt*v+  // either scaling or scaled addition (scalar = timestep)
            vScalAcc(positions[i], velocityAfter[i],timeStep);
            // v = avg(v+,v-) // scaled addition (scalar=0.5)
            vINTACC(velocities[i], velocityAfter[i],velocityBefore[i]);

            velocityMax = Math.max(velocityMax, norm2(velocities[i])); // For use in dynamic time-stepping/smoothing length.

            // v- = v+ // just switch the pointers
            vAssign(velocityBefore[i],velocityAfter[i]);
        }
    }

    /**
     * Determines a smoothing length that ensures "particleRequirement" particles fall within
     * the smoothing length used for the smoothing kernels.
     */
    private static float determineSmoothingLength(int particleRequirement){
        // PI * smoothingLength^2 / particleArea = particleRequirement
        // smoothingLength = sqrt(particleArea * particleRequirement / PI)
        // smoothingLength = sqrt(PI * r * r * particleRequirement / PI)
        // smoothingLength = particleRadius * sqrt(particleRequirement)

        float base = (particleRadius*particleRequirement);
        return base;
    }

    private static void vScalIP(float[] vec, float scalar){
        vec[0] *= scalar;
        vec[1] *= scalar;
        //vec[2] *= scalar;
    }
    private static float getParticleExtent(){
        float[] positionAverage = new float[2]; // Find the average position.
        for (int i = 0; i < N_particles; i++) {
              vaddIP(positionAverage,positions[i]);
        }
        vScalIP(positionAverage,1F/N_particles);

        // Find the average distance from this position.
        float diffSum = 0f;
        for (int i = 0; i < N_particles; i++) {
            diffSum += norm2(diff(positions[i],positionAverage));
        }
        return diffSum / N_particles;
    }

    private static float maximumTimeStep = 4F;
    /*public static void main(String[] args) throws InterruptedException {
        initParameters();
        initGUI();

        ExecutorService threadPool = Executors.newFixedThreadPool(32);
        initMultithreading();

        long realtimeLast = System.currentTimeMillis();

        referenceDensity = densities[0];
        graphicsPanel.setReferenceDensity(referenceDensity);

        for (int t = 0; t < maxTimeSteps; t++) {
            // Keep track of real times for calculating performance.
            long realtimeNow = System.currentTimeMillis();
            long realTimeDiff = realtimeNow -realtimeLast;
            realtimeLast = realtimeNow;

            // Dynamic updates.
            smoothingLength = earthRadius / 25;
            //smoothingLength = determineSmoothingLength(4);
            softeningLength = smoothingLength;

            System.out.println("-----------------");
            System.out.println("t="+t);
            // Time-integration, calculates velocities and positions from acceleration.
            if(t == 0){
                timeStep = maximumTimeStep;
            }
            if(t > 0) {
                if(t == 1){
                    // One-time calculations on t=1, since we may lack the information on t=0.
                    referenceDensity = densities[0];
                    graphicsPanel.setReferenceDensity(referenceDensity);
                }
                integrate();

                graphicsPanel.update(smoothingLength,realTimeDiff);
                System.out.println("v_max=" + velocityMax);
                printSimulationStats();
                timeStep = Math.min((smoothingLength / velocityMax), maximumTimeStep);
                System.out.println("Δt="+timeStep);
                System.out.println("Δt/t_max="+((timeStep/maximumTimeStep)*100)+"%");
            }

            // Update densities, pressures and accelerations.
            calculateDensities(threadPool);
            //calculateSolidPressures();
            //calculatePressures(4182,8F/6,295);
            updateAccelerations(threadPool);
            //updateAccelerationsMK2();

            graphicsPanel.update(smoothingLength,realTimeDiff);
            printSimulationStats();
        }

        threadPool.shutdown();
        threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    }*/
}
