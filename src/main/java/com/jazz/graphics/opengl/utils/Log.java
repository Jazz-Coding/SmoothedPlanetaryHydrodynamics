package com.jazz.graphics.opengl.utils;

public class Log {
    private LogLevel logLevel;

    public Log(LogLevel logLevel) {
        this.logLevel = logLevel;
    }

    public Log() {
        this.logLevel=LogLevel.DEBUG;
    }

    public void setLogLevel(LogLevel logLevel) {
        this.logLevel = logLevel;
    }

    public void print(String line, LogLevel level){
        if(level == logLevel || logLevel==LogLevel.ALL){
            System.out.println(line);
        }
    }
}
