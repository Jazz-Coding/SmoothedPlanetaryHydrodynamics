package com.jazz.graphics;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

/**
 * Debugging GUI - 2D graphics with CPU computation
 */
public class BasicPanel extends JPanel implements MouseWheelListener, MouseListener {
    private float[][] positions;
    private float[][] accelerations;
    private float[][] velocities;

    private float[] densities;
    private float referenceDensity = 1f;
    private float[] pressures;

    private float smoothLength;
    private long frameTime;

    private float sceneWidth;
    private float particleWidth;
    private float downscaleFactor;
    private float particleWidthDownscaled;

    private float dragOffsetX = 0;
    private float dragOffsetY = 0;

    public BasicPanel(float[][] positions,
                      float[][] accelerations,
                      float[][] velocities,
                      float[] densities,
                      float[] pressures,
                      float sceneWidth, float particleWidth) {
        this.positions = positions;
        this.accelerations = accelerations;
        this.velocities = velocities;
        this.densities = densities;
        this.pressures = pressures;

        this.sceneWidth = sceneWidth;
        this.particleWidth = particleWidth;

        this.downscaleFactor = sceneWidth / 512;
        this.particleWidthDownscaled = particleWidth / downscaleFactor;

        setBackground(Color.BLACK);
        addMouseWheelListener(this);
        addMouseListener(this);

        new Thread(() -> {
            while (true){
                try {
                    Thread.sleep(1000/175);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                repaint();
            }
        }).start();
    }

    private float lastColor = 0f;

    private void doDrawing(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
        /*g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);*/

        int w = getWidth();
        int h = getHeight();

        /*System.out.println(sceneWidth);
        System.out.println(particleWidth);*/

        float accMax = 0f;
        float densMax = 0f;
        float pressMax = 0f;
        float vMax = 0f;


        for (int i = 0; i < accelerations.length; i++) {
            float currentAcceleration = (float) Math.sqrt(accelerations[i][0]*accelerations[i][0] +
                    accelerations[i][1]*accelerations[i][1]);
            float currentVelocity = (float) Math.sqrt(velocities[i][0]*velocities[i][0] +
                    velocities[i][1]*velocities[i][1]);
            accMax = Math.max(accMax, currentAcceleration);
            vMax = Math.max(vMax, currentVelocity);

            densMax = Math.max(densMax, densities[i]);
            pressMax = Math.max(pressMax, pressures[i]);
        }

        g2d.setPaint(Color.WHITE);
        g2d.setFont(new Font( "SansSerif", Font.PLAIN, 18));
        g2d.drawString("Max. Acceleration: " + accMax,5,15);
        g2d.drawString("Max. Density: " + densMax,5,35);
        g2d.drawString("d/d_0: " + (densMax/referenceDensity),5,55);
        g2d.drawString("Max. Pressure: " + pressMax, 5 ,77);
        g2d.drawString("N_particles: " + positions.length,5,105);

        int fps = (int) (1000 / (frameTime+1));
        g2d.drawString("FPS: " + fps, 5 ,135);


        int smoothSize = Math.round(smoothLength/downscaleFactor);
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);
        smoothSize++;
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);
        smoothSize++;
        g2d.drawOval(512-smoothSize/2,512-smoothSize/2, smoothSize, smoothSize);

        for (int i = 0; i < positions.length; i++) {
            float x = (positions[i][0] / downscaleFactor);
            float y = (positions[i][1] / downscaleFactor);

            float acc = (float) Math.sqrt(accelerations[i][0] * accelerations[i][0] +
                                          accelerations[i][1] * accelerations[i][1]);

            float v = (float) Math.sqrt(velocities[i][0]*velocities[i][0] +
                    velocities[i][1]*velocities[i][1]);
            float dens = densities[i];
            float press = pressures[i];

            float newColorFactor = Math.min(1.0f, Math.max(0f,v/vMax));

            Color c = new Color(newColorFactor,0,0.5f);

            int realX = (int) (x + w / 2) + (int) dragOffsetX/2;
            int realY = (int) (y + h / 2) + (int) dragOffsetY/2;


            //System.out.println(acc / accColorDownscale);
            g2d.setPaint(c);
            //g2d.drawLine(realX, realY, realX, realY);
            g2d.fillOval(realX, realY, (int) Math.min(4,particleWidthDownscaled), (int) Math.min(4,particleWidthDownscaled));
        }
        /*for (float[] position : positions) {
            float x = (position[0] / downscaleFactor);
            float y = (position[1] / downscaleFactor);

            int realX = (int) (x + w / 2);
            int realY = (int) (y + h / 2);

            //g2d.drawLine(realX, realY, realX, realY);
            g2d.fillOval(realX, realY, (int) particleWidthDownscaled, (int) particleWidthDownscaled);
        }*/
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        doDrawing(g);
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        this.sceneWidth += e.getWheelRotation()*1e6;
        this.downscaleFactor = sceneWidth / 512;
        this.particleWidthDownscaled = particleWidth / downscaleFactor;
    }

    public void update(float smoothLength, long frameTime) {
        this.smoothLength = smoothLength;
        this.frameTime = frameTime;
        //System.arraycopy(positions,0,this.positions,0,positions.length);
        //repaint();
    }

    public void setReferenceDensity(float referenceDensity) {
        this.referenceDensity = referenceDensity;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        int clickX = e.getX();
        int clickY = e.getY();

        this.dragOffsetX = clickX;
        this.dragOffsetY = -clickY;
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
