#version 330 core
flat in ivec3 intcolor;
//flat in int r;
in float gl_ClipDistance[6];
out ivec4 FragColor;
//out int FragColor;
uniform mat4 MVP, itMVP;
uniform vec3 vmin, vmax;
uniform vec2 viewport;
uniform float radius, scale;
const float lighting=0.75;
void main() {
    // Вроде и без этого должно раьотать, но как-то на amd не очень
    //for(int i=0; i<6; i++){
    //    if (gl_ClipDistance[i]<=0) discard;
    //}
    const float camera_light = 0.95;
    const float vertical_light = 1.0;
    vec2 shift = gl_PointCoord.xy -vec2(0.5,0.5);
    float d = dot(shift,shift);
    if (d>= 0.25) discard;
    float dz = sqrt(1. - 4.*d/256.);
    FragColor = ivec4(intcolor,127);
    //FragColor = r;
    gl_FragDepth =gl_FragCoord.z-radius*dz/distance(vmax,vmin);
}
