#version 330 core
in vec3 coord;//, normal;
in float color;
uniform mat4 MVP, itMVP;
uniform vec3 vmax, vmin, minmaxmul;
uniform float scale;
out float gl_ClipDistance[6];
out float c;
//flat out ivec3 intcolor;
void main() {
    for(int i =0; i<3; i++){
        gl_ClipDistance[i]=coord[i]- vmin[i];
        gl_ClipDistance[i+3]=vmax[i] -coord[i];
    }
    vec4 res = MVP*vec4(coord,1.0);
    vec3 center = (vmin+vmax)*0.5;
    res.z = 2*((MVP*vec4(center,1.)).z - res.z)*scale/(distance(vmax,vmin)); // in [-1, 1]
    c = (color-minmaxmul.x)/(minmaxmul.y - minmaxmul.x);
    //intcolor = ivec3(r,g,b);
    gl_Position = res;
}
