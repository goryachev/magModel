#version 330 core
in vec3 coord, normal;
in float color;
uniform vec3 vmin, vmax;
uniform mat4 MVP, itMVP;
uniform vec3 minmaxmul;
uniform vec2 viewport;
uniform float scale;
out float gl_ClipDistance[6];
out float c, n;
const float lighting=0.75;
void main() {
    const float camera_light = 0.95;
    const float vertical_light = 0.05;
    vec4 res = MVP*vec4(coord,1.0); // It's recomended to use right multiplication, I wonder why
    for(int i =0; i<3; i++){
        gl_ClipDistance[i]=coord[i]- vmin[i];
        gl_ClipDistance[i+3]=vmax[i] -coord[i];
    }
    vec3 center = (vmin+vmax)*0.5;

    res.z = 2*((MVP*vec4(center,1.)).z-res.z)*scale/(distance(vmax,vmin)); // in [-1, 1]
    n = lighting*abs(camera_light*(itMVP*vec4(normal,0.)).z / scale +
                (normal.x+normal.y+normal.z)/3.*vertical_light)/(camera_light+vertical_light);
    c = (color-minmaxmul.x)*minmaxmul.z;
    gl_Position = res;
}
