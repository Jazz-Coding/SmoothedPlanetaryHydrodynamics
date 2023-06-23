package com.jazz.graphics;

import com.jazz.gpulib.VectorMath;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Hashtable;

import static com.jazz.gpulib.SPH_GPU.*;

/**
 * Debugging GUI - 2D graphics with GPU computation
 */
public class BasicPanelGPU extends JPanel implements MouseWheelListener, MouseListener{
    private float particleMass;

    private float smoothLength;
    private long frameTime;

    private int windowWidth;
    private float sceneWidth;

    public Coloring particleColoringMethod = Coloring.ACCELERATION;

    private int pixelsPerUnitBase;
    private int pixelsPerUnit;

    private Point mousePt;
    private float centreX = 0;
    private float centreY = 0;

    private int targetFPS = 60;
    private boolean VSYNC = false;
    private JSlider timeDial = new JSlider(JSlider.HORIZONTAL, 0, 1000, 0);

    public BasicPanelGPU(int windowWidth, float sceneWidth, float particleMass, boolean VSYNC) {
        this.particleMass=particleMass;
        this.VSYNC = VSYNC;


        this.windowWidth = windowWidth;
        this.sceneWidth = sceneWidth;

        this.pixelsPerUnit = (int) Math.ceil(windowWidth / sceneWidth);
        this.pixelsPerUnitBase = pixelsPerUnit;

        setBackground(Color.BLACK);

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException |
                 UnsupportedLookAndFeelException e) {
            throw new RuntimeException(e);
        }

        timeDial.setBackground(Color.BLACK);
        timeDial.setForeground(Color.WHITE);
        timeDial.setPreferredSize(new Dimension(500,50));
        timeDial.setPaintTicks(true);
        timeDial.setPaintLabels(true);
        timeDial.setFocusable(false);
        Hashtable<Integer, JLabel> labelTable = new Hashtable();
        JLabel labelStart = new JLabel("0");
        labelStart.setBackground(Color.BLACK);
        labelStart.setForeground(Color.WHITE);

        labelTable.put(0, labelStart);
        JLabel labelEnd = new JLabel("1.0");
        labelEnd.setBackground(Color.BLACK);
        labelEnd.setForeground(Color.WHITE);
        labelTable.put(1000, labelEnd);
        timeDial.setLabelTable(labelTable);

        this.add(timeDial);

        addMouseWheelListener(this);
        addMouseListener(this);
        this.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                mousePt = e.getPoint();
                repaint();
            }
        });
        this.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int dx = e.getX() - mousePt.x;
                int dy = e.getY() - mousePt.y;
                centreX += dx;
                centreY += dy;
                mousePt = e.getPoint();
                //repaint();
            }
        });

        //updateSimulationData();
        if(!VSYNC) {
            long frametime = 1000 / targetFPS;
            new Thread(() -> {
                while (true) {
                    try {
                        Thread.sleep(frametime);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    repaint();
                }
            }).start();
        }
    }

    private static int debugLineLastX = 5;
    private static int debugLineLastY = 25;

    private static Font titleFont = new Font("SansSerif",Font.BOLD,24);
    private static Font normalFont = new Font("SansSerif",Font.PLAIN,18);

    private void drawNormalText(Graphics2D g2d, String text){
        g2d.setPaint(Color.WHITE);
        g2d.setFont(normalFont);
        g2d.drawString(text,debugLineLastX,debugLineLastY);
        debugLineLastY+=18;
    }
    private void drawTitleText(Graphics2D g2d, String text){
        g2d.setPaint(Color.WHITE);
        g2d.setFont(titleFont);
        g2d.drawString(text,debugLineLastX,debugLineLastY);
        debugLineLastY+=24;
    }
    private void renderSimulationStatistics(Graphics2D g2d){
        debugLineLastX = 5;
        debugLineLastY = 25;

        Color originalColor = g2d.getColor();
        //float[] netMomentum = PhysicsTests.netMomentum(particleMass, velocities);
        //float avgNetMomentum = VectorMath.norm2(netMomentum);

        drawTitleText(g2d,"----Physics Statistics (avg/max)----");
        drawNormalText(g2d,"[N_particles]" + N_PARTICLES);
        drawNormalText(g2d, String.format("[Particle Extent] %f/%f", averageParticleExtent,greatestParticleExtent));
        drawNormalText(g2d, String.format("[Velocity] %f/%f", averageVelocity,highestVelocity));
        drawNormalText(g2d, String.format("[Acceleration] %f/%f", averageAcceleration,highestAcceleration));
        drawNormalText(g2d, String.format("[Force] %f/%f", averageForce,highestForce));
        drawNormalText(g2d, String.format("[Density] %f/%f", averageDensity,highestDensity));
        //drawNormalText(g2d,String.format("[Momentum (net)] %f",avgNetMomentum));
        debugLineLastY+=25;

        drawTitleText(g2d,"----Performance Statistics----");
        int fps = (int) (1000 / (frameTime+1));
        drawNormalText(g2d, String.format("[FPS] %d", fps));



        float interactionsDensities = (N_PARTICLES*1F) + (float) (Math.pow(N_PARTICLES,2)*16);
        float interactionsPressures = (N_PARTICLES*2F);
        float interactionsAccelerations = (N_PARTICLES*14F) + (float) (Math.pow(N_PARTICLES,2)*33);
        float interactions = interactionsDensities + interactionsPressures + interactionsAccelerations;
        float tflops = (interactions / frameTime) / 1e9F;

        drawNormalText(g2d,String.format("[TFLOPs] %.2f (%.2f%% of theoretical maximum)",tflops,(100*(tflops/80F))));
        g2d.setPaint(originalColor);
    }

    /**
     * Renders a representation of the current smoothing length.
     */
    private void renderDebugData(Graphics2D g2d){
        Color originalColor = g2d.getColor();
        g2d.setPaint(Color.WHITE);

        int smoothSize = Math.round(smoothLength * pixelsPerUnit);
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);
        smoothSize++;
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);
        smoothSize++;
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);

        g2d.drawString("Pixels per Earth: ",15,300);
        g2d.fillRect(15,350, pixelsPerUnit *2,4);
        g2d.setPaint(originalColor);
    }



    public void setParticleColoringMethod(Coloring particleColoringMethod) {
        this.particleColoringMethod = particleColoringMethod;
    }

    private Color colorParticle(int particleIndex){
        float variable = 1F;

        Color c1 = Color.RED; // Highest
        Color c2 = Color.BLUE; // Lowest

        switch (particleColoringMethod){
            case SPEED -> {variable = VectorMath.norm2(velocities[particleIndex]) / highestVelocity; c1 = Color.MAGENTA; c2 = Color.PINK;}
            case ACCELERATION -> {variable = VectorMath.norm2(accelerations[particleIndex]) / highestAcceleration; c1 = Color.RED; c2 = Color.BLUE;}
            case DENSITY -> {variable = densities[particleIndex] / highestDensity; c1 = Color.WHITE; c2 = Color.BLACK;}
        }

        // Define the smallest to be one hue, and the greatest to be another.
        float[] HSV_RED = new float[3];
        float[] HSV_BLUE = new float[3];

        Color.RGBtoHSB(c1.getRed(), c1.getGreen(), c1.getBlue(),HSV_RED);
        Color.RGBtoHSB(c2.getRed(), c2.getGreen(), c2.getBlue(),HSV_BLUE);

        float ratio = 1F-variable;
        float[] BLEND = new float[]{HSV_RED[0]*variable+HSV_BLUE[0]*ratio,
                                    HSV_RED[1]*variable+HSV_BLUE[1]*ratio,
                                    HSV_RED[2]*variable+HSV_BLUE[2]*ratio};
        return Color.getHSBColor(BLEND[0],BLEND[1],BLEND[2]);
    }

    private void doDrawing(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);

        renderSimulationStatistics(g2d);
        renderDebugData(g2d);

        int zoomLevel = Math.max(1,pixelsPerUnit / pixelsPerUnitBase);

        for (int i = 0; i < positions.length; i++) {
            float x = positions[i][0];//+sceneWidth/2; // Centres the positions.
            float y = positions[i][1];//+sceneWidth/2;

            int xPix = (int) (x * pixelsPerUnit);
            int yPix = (int) (y * pixelsPerUnit);

            xPix += windowWidth/(2)+centreX;
            yPix += windowWidth/(2)+centreY;

            //System.out.printf("Rendering particle from (%f,%f) at (%d,%d). PPM=%d%n", x,y, xPix, yPix,pixelsPerMetre);
            g2d.setPaint(colorParticle(i));
            g2d.fillOval(xPix, yPix, zoomLevel, zoomLevel);
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        doDrawing(g);
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        boolean forward;
        int wheelRotation = e.getWheelRotation();
        forward = wheelRotation  <= 0;
        if(forward) {
            this.pixelsPerUnit += 5;
        } else {
            this.pixelsPerUnit -= 5;
            if(this.pixelsPerUnit <=0){
                this.pixelsPerUnit = 1;
            }
        }
    }

    public void update(float smoothLength, long frameTime) {
        this.smoothLength = smoothLength;
        this.frameTime = frameTime;
        if(VSYNC) {
            repaint();
        }
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {

    }

    @Override
    public void mouseReleased(MouseEvent e) {

    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }
}
