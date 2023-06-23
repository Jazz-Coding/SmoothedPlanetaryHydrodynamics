package com.jazz.utils;

/**
 * Helper for timing execution, enables performance evaluation and framerate measuring.
 */
public class Stopwatch {
    private long begin;
    private long end;

    public void start(){
        this.begin = System.nanoTime();
    }

    public void stop(){
        this.end = System.nanoTime();
    }

    public float durationMS(){
        return (end-begin)/1e6F;
    }
}
