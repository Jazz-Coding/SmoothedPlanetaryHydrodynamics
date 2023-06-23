#version 330

in vec4 decided_color; // Color from the vertex shader.
in vec3 directionToLight;
flat in vec3 interpolated_normal; // Normal from vertex shader
out vec4 outputColor; // RGBA color component.

void main() { // The fragment shader now receives a "pixel" from the rasterizer, and can decide on how to display it in the actual window.
        float dotProd = dot(interpolated_normal.xzy,directionToLight);
        float falloff = 0.5;
        float lightIntensity = max(pow(dotProd, falloff), 0.3);
        outputColor = lightIntensity * decided_color;//timeOffset * decided_color;
}