package com.jazz;

import com.jazz.gpulib.MatrixMath;
import com.jazz.gpulib.SPH_GPU;
import com.jazz.graphics.opengl.Camera;
import com.jazz.graphics.opengl.Shader;
import com.jazz.graphics.opengl.primitives.Mesh;
import com.jazz.graphics.opengl.primitives.MeshBuilder;
import com.jazz.graphics.opengl.primitives.simple.Vertex;
import com.jazz.graphics.opengl.utils.STLReader;
import org.joml.Vector3f;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL20;
import org.lwjgl.system.MemoryStack;

import java.awt.*;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;

/**
 * The main class for the GPU implementation of the simulation.
 * Initializes and dispatches to the GPU backend.
 */
public class Main {
    // The window handle
    private long window;

    // The shading program handle.
    private int sceneProgram;
    private int UIprogram;

    private int windowWidth=1920;
    private int windowHeight=1080;

    // User controls.
    private boolean isRightButtonPressed;
    private boolean isShiftPressed;

    private float mouseX;
    private float mouseY;


    public void run() {
        System.out.println("Running LWJGL " + Version.getVersion() + "!");

        init();
        loop();

        // Free the window callbacks and destroy the window
        glfwFreeCallbacks(window);
        glfwDestroyWindow(window);

        // Terminate GLFW and free the error callback
        glfwTerminate();
        glfwSetErrorCallback(null).free();
    }

    private void init() {
        // Setup an error callback.
        GLFWErrorCallback.createPrint(System.err).set();

        // Initialize GLFW. Most GLFW functions will not work before doing this.
        if (!glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }

        // Configure GLFW
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // The window will stay hidden after creation
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // The window will be resizable

        // Create the window
        window = glfwCreateWindow(windowWidth, windowHeight, "SPH", NULL, NULL);
        if ( window == NULL ) {
            throw new RuntimeException("Failed to create the GLFW window");
        }

        // Keybind setup.
        glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
            if ( key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE ) {
                glfwSetWindowShouldClose(window, true); // We will detect this in the rendering loop
            } else if(key == GLFW_KEY_0 && action == GLFW_RELEASE){
                SPH_GPU.zeroVelocities();
            } else if(key == GLFW_KEY_UP && action == GLFW_RELEASE){
                timestep *=1.25F;
            } else if(key == GLFW_KEY_DOWN && action == GLFW_RELEASE){
                timestep *=0.75F;
            } else if(key == GLFW_KEY_LEFT_SHIFT){
                isShiftPressed = (action != GLFW.GLFW_RELEASE);
            } else if(key == GLFW_KEY_PAUSE && action == GLFW_RELEASE){
                paused = !paused;
            } else if(key == GLFW_KEY_BACKSPACE && action == GLFW_RELEASE){
                timestep*=-1F;
            }
        });

        // Setup a resize callback. This rescales the viewport when the window changes, stretching the displayed image.
        glfwSetWindowSizeCallback(window, new GLFWWindowSizeCallback() {
            @Override
            public void invoke(long windowHandle, int width, int height) {
                windowWidth = width;
                windowHeight = height;

                updatePerspective();
                glViewport(0,0,width,height);
            }
        });


        // Set the mouse button callback
        GLFW.glfwSetMouseButtonCallback(window, new GLFWMouseButtonCallback() {
            @Override
            public void invoke(long window, int button, int action, int mods) {
                if (button == GLFW.GLFW_MOUSE_BUTTON_RIGHT) {
                    isRightButtonPressed = (action != GLFW.GLFW_RELEASE);
                }
            }
        });

        // Set the scroll callback
        GLFW.glfwSetScrollCallback(window, new GLFWScrollCallback() {
            @Override
            public void invoke(long window, double ignored, double yoffset) {
                camera.zoom((float) (yoffset/10));
            }
        });

        // Set the cursor position callback
        GLFW.glfwSetCursorPosCallback(window, new GLFWCursorPosCallback() {
            @Override
            public void invoke(long window, double xpos, double ypos) {
                float dx = (float) (xpos-mouseX);
                float dy = (float) (ypos-mouseY);

                if (isRightButtonPressed) {
                    if(isShiftPressed){
                        // Translate.
                        camera.translate(dx/50,dy/50);
                    } else {
                        // Orbit.
                        camera.orbitTarget(dx,dy);
                    }
                }
                // Update the stored mouse position
                mouseX = (float) xpos;
                mouseY = (float) ypos;
            }
        });

        // Get the thread stack and push a new frame
        try ( MemoryStack stack = stackPush() ) {
            IntBuffer pWidth = stack.mallocInt(1); // int*
            IntBuffer pHeight = stack.mallocInt(1); // int*

            // Get the window size passed to glfwCreateWindow
            glfwGetWindowSize(window, pWidth, pHeight);

            // Get the resolution of the primary monitor
            GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());

            // Center the window
            glfwSetWindowPos(
                    window,
                    (vidmode.width() - pWidth.get(0)) / 2,
                    (vidmode.height() - pHeight.get(0)) / 2
            );
        } // the stack frame is popped automatically

        // Make the OpenGL context current
        glfwMakeContextCurrent(window);

        // Enable v-sync
        glfwSwapInterval(1);

        // Make the window visible
        glfwShowWindow(window);
    }

    private int buildShaderProgram(String vertexShaderSourcePath,
                                   String fragmentShaderSourcePath){
        int program = glCreateProgram();

        // Compile the shaders.
        Shader sceneVertexShader = new Shader(vertexShaderSourcePath);
        Shader sceneFragmentShader = new Shader(fragmentShaderSourcePath);

        sceneVertexShader.attach(program);
        sceneFragmentShader.attach(program);

        // Link the full program.
        glLinkProgram(program);
        glValidateProgram(program);

        if(program==0){
            throw new RuntimeException("Shader program build failed!");
        }

        // Free unused resources.
        sceneVertexShader.free(program);
        sceneFragmentShader.free(program);

        return program;
    }
    private void initShaders(){
        sceneProgram = buildShaderProgram("shaders/vertex.vs","shaders/fragment.fs");
        UIprogram = buildShaderProgram("shaders/ui_vertex.vs","shaders/ui_fragment.fs");
        glUseProgram(sceneProgram);
    }

    private int modelMatrixUniformHandle;
    private int viewMatrixUniformHandle;
    private int perspectiveMatrixUniformHandle;

    private void loadMVPUniforms(){
        modelMatrixUniformHandle = glGetUniformLocation(sceneProgram, "modelMatrix");
        viewMatrixUniformHandle = glGetUniformLocation(sceneProgram, "viewMatrix");
        perspectiveMatrixUniformHandle = glGetUniformLocation(sceneProgram, "perspectiveMatrix");
    }

    private void updatePerspective(){
        GL20.glUniformMatrix4fv(viewMatrixUniformHandle, false,camera.getViewMatrix().get(new float[16]));
        GL20.glUniformMatrix4fv(perspectiveMatrixUniformHandle, false,camera.getPerspectiveMatrix().get(new float[16]));
    }

    private List<float[]> randomXYZPositions(int n, float lowerBound, float upperBound){
        ThreadLocalRandom RNG = ThreadLocalRandom.current();
        List<float[]> positions = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            positions.add(new float[]{
                    RNG.nextFloat(lowerBound,upperBound),
                    RNG.nextFloat(lowerBound,upperBound),
                    RNG.nextFloat(lowerBound,upperBound),
                    1F});
        }

        return positions;
    }

    private void pre_init(){
        initShaders();
        loadMVPUniforms();

        // Set the clear colors and depth.
        glClearDepth(1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // Enable culling.
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW); // Front facing triangles are ordered clockwise (CW). So cull anti-clockwise ones.

        glEnable(GL_DEPTH_TEST);
        glDepthMask(true);
        glDepthFunc(GL_LEQUAL);
        glDepthRange(0.0f, 1.0f);
    }

    private int setUniform(String name, float[] values){
        int handle = glGetUniformLocation(sceneProgram, name);
        switch (values.length){
            case 2 -> glUniform2f(handle,values[0],values[1]);
            case 3 -> glUniform3f(handle,values[0],values[1], values[2]);
            case 4 -> glUniform4f(handle,values[0],values[1], values[2], values[3]);
        }
        return handle;
    }

    private void pre_render(){
        updatePerspective();
        Vector3f lightPos = camera.getCameraPos();
        setUniform("lightPos",new float[]{lightPos.x,lightPos.y,lightPos.z,1});
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
    }

    private void post_render(){
        // Stuff afterwards.
        glfwSwapBuffers(window); // swap the color buffers

        // Poll for window events. The key callback above will only be
        // invoked during this call.
        glfwPollEvents();
    }


    private Camera camera;
    private boolean paused;
    private float timestep=10F;

    private void optimizeBlockSize(int N_particles){
        int SMs = 128; // Hardware number.
        int estimated = Math.min(1024,N_particles / SMs); // Hardware limit is 1024.

        // Pick the nearest multiple of 32.
        int count = 32;
        while (count < estimated){
            count+=32;
        }

        MatrixMath.THREADS_PER_BLOCK = count;
    }

    private void loop() {
        GL.createCapabilities();
        pre_init();

        List<Vertex> icoVerts = STLReader.readSTLVerts("meshes/icosphere.stl",Color.WHITE);
        List<float[]> vertexNormals = STLReader.readSTLNormals("meshes/icosphere.stl");

        int count = 1024*32;
        optimizeBlockSize(count);
        System.out.println("THREADS PER BLOCK: " + MatrixMath.THREADS_PER_BLOCK);


        // Create "count" copies of the normals to work with the instanced rendering.
        List<float[]> repeatedNormals = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            repeatedNormals.addAll(vertexNormals);
        }

        float pscale = 0.1F;
        System.out.println(count);
        Mesh icosphere = new MeshBuilder().
                collectVerticesWithInstanceOffsetsAndNormals(icoVerts,randomXYZPositions(count,-50,50),repeatedNormals)
                .setScale(pscale,pscale,pscale)
                .build();

        System.out.println("Initializing SPH simulation...");
        SPH_GPU.init(count);
        SPH_GPU.prepCudaInterop(icosphere.get_VBO());

        camera = new Camera(new Vector3f(0,0,-10),60);

        // Rendering loop.
        while ( !glfwWindowShouldClose(window) ) {
            pre_render();

            // Simulation call.
            if(!paused) {
                SPH_GPU.doTimestep(timestep);

                // Copy to GUI.
                SPH_GPU.GPU_GUI_UPDATE();
            }

            // Render geometry.
            icosphere.drawInstances(modelMatrixUniformHandle,count);

            post_render();
        }
    }

    public static void main(String[] args) {
        new Main().run();
    }
}
