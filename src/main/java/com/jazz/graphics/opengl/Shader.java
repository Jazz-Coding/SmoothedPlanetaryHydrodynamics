package com.jazz.graphics.opengl;

import com.jazz.graphics.opengl.utils.FileUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL20;

import static org.lwjgl.opengl.GL20.*;

/**
 * Loads, compiles and links vertex and fragment shaders written in GLSL (GL shading language).
 */
public class Shader {
    private int identifier;

    public Shader(String sourceCodePath) {
        int inferredType;

        if(sourceCodePath.endsWith(".vs")){
            inferredType = GL_VERTEX_SHADER;
        } else {
            inferredType = GL_FRAGMENT_SHADER;
        }

        String code = FileUtils.fileToString(sourceCodePath);
        this.identifier = compileShader(code, inferredType);
    }

    private static int compileShader(String source, int type){
        int shader = glCreateShader(type);
        if(shader==0){
            throw new RuntimeException("Shader creation failed.");
        }
        glShaderSource(shader,source);
        glCompileShader(shader);

        // Check for errors during compilation
        if (GL20.glGetShaderi(shader, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
            throw new RuntimeException("Error creating shader\n" + GL20.glGetShaderInfoLog(shader, GL20.glGetShaderi(shader, GL20.GL_INFO_LOG_LENGTH)));
        }
        return shader;
    }

    public void attach(int program){
        glAttachShader(program, identifier);
    }

    public void free(int program){
        glDetachShader(program,identifier);
        glDeleteShader(identifier);
    }
}
