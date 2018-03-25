#include <iostream>
#include <iomanip>
//#include <string>
#include <cstring>
#include <chrono>
#include "viewer_template.hpp"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
//#include "frontview.hpp"
const double much_enough = 1e3;
//------------------------------------------------------------------------------
void checkOpenGLerror()
{
    GLenum errCode;
    if((errCode=glGetError()) != GL_NO_ERROR)
        std::cout << "OpenGl error! - " << gluErrorString(errCode)<<std::endl;
}
//--------------------------------------------------------------------------------
void Viewer::togglewire() {
    set_wire(!wire);
}
void Viewer::set_wire(bool tf){
    wire = tf;
    glPolygonMode(GL_FRONT_AND_BACK, wire ? GL_LINE : GL_FILL);
}
//--------------------------------------------------------------------------------
//Повороты
//--------------------------------------------------------------------------------
glm::quat Viewer::get_rot_tmp() const {
        glm::quat quatx = glm::angleAxis(rotx*Sens, glm::vec3(1.f,0.f,0.f));
        return glm::angleAxis(roty*Sens, quatx * glm::vec3(0.f,1.f,0.f) /* quatx*/) * // y is inversed
            quatx ;
}
//--------------------------------------------------------------------------------
void Viewer::mouse_click(int button, int state, int x, int y){
    if (button ==GLUT_LEFT_BUTTON && state == GLUT_DOWN){
        tr0.x = ((float)x)/min_size*2;
        tr0.y = ((float)y)/min_size*2;
        left_click = true;
    }
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP){
        auto rot_tmp = get_rot_tmp();
        pos -= glm::inverse(orient*rot_tmp)* glm::vec3(tr*scale,0.f);
        tr = glm::vec2(0.f);
        left_click = false;
    }
    if (button ==GLUT_RIGHT_BUTTON && state == GLUT_DOWN){
        ox = ((float)x)/min_size*2;
        oy = ((float)y)/min_size*2;
        right_click = true;
    }
    if (button ==GLUT_RIGHT_BUTTON && state == GLUT_UP){
        auto rot_tmp = get_rot_tmp();
        orient = rot_tmp * orient;
        rotx = 0.f; roty = 0.f;
        ox = 0.f; oy = 0.f;
        right_click = false;
    }
    if (button ==3&& state == GLUT_DOWN){
        //нужно подумать
        //pos += glm::vec3(0.f,0.f,scale*0.1);
        scale *= 0.9;
        reshape(width,height);
    } else if (button == 4&& state == GLUT_DOWN){
        scale /=0.9;
        //pos -= glm::vec3(0.f,0.f,scale*0.1);
        reshape(width,height);
    }
}
//--------------------------------------------------------------------------------
void Viewer::drag(int x, int y){
    if (left_click){
        float px = ((float)x)/min_size*2;
        float py = ((float)y)/min_size*2;
        tr.x = px -tr0.x;
        tr.y = -(py -tr0.y);//z is inverted
        //calc_mvp();
    }
    if (right_click){
        float px = ((float)x)/min_size*2;
        float py = ((float)y)/min_size*2;
        //rotx=0;
        rotx = glm::degrees((py - oy) * 4*M_PI/180.f);
        roty = glm::degrees((px - ox) * 4*M_PI/180.f);
        //rotate(ox,py,px);
        //calc_mvp();
    }
    glutPostRedisplay();
}
void Viewer::GL_init(){
    checkOpenGLerror();
    //printf("init begin\n");
    glewExperimental=GL_TRUE;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(-1.,1.);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_CLIP_DISTANCE0);
    glEnable(GL_CLIP_DISTANCE1);
    glEnable(GL_CLIP_DISTANCE2);
    glEnable(GL_CLIP_DISTANCE3);
    glEnable(GL_CLIP_DISTANCE4);
    glEnable(GL_CLIP_DISTANCE5);
    //glEnable( GL_BLEND );
    //glBlendFunc(GL_ONE, GL_ONE);
    glewInit();
    checkOpenGLerror();
    //printf("init end\n");
}
//--------------------------------------------------------------------------------
//i==0,1,2 — min x,y,z
//i=3,4,5 — max ,x,y,z
void Viewer::clip_plane_move(float shift, int num){
    if (num<3) min[num] += shift*scale;
    else max[num%3] += shift*scale;
}
//--------------------------------------------------------------------------------
Viewer::Viewer(){
    rotx=0.0f; roty=0.0f;
    ox = 0.f; oy = 0.f;
    scale = 1.f;
    pos = glm::vec3(0.f); pos.z = 0.f;
    tr = glm::vec2(0.f); tr0 = glm::vec2(0.f);
    min=glm::vec3(0.); max=glm::vec3(0.);
    left_click = false; right_click = false;
    wire = false;
    width = 900; height = 900;
    //orient =glm::quat(glm::vec3(0.,M_PI_2,-M_PI_2));
    //orient =glm::quat(glm::vec3(-M_PI_2,-M_PI_2,0.f)); // real order x z x
    orient =glm::quat(glm::vec3(-M_PI_2,-M_PI_2,0.f)); // real order x z x
    ort = glm::ortho(-1,1,-1,1,1,-1); //z is also inverted by glm
    background = glm::vec3(1.f,1.f,1.f);
    axis_sw = false;
    checkOpenGLerror();
}
//--------------------------------------------------------------------------------
void Viewer::get_background(float &r, float &g, float &b){
    r = background.x;
    g = background.y;
    b = background.z;
}
//--------------------------------------------------------------------------------
void Viewer::set_background(float r, float g, float b){
    background = glm::vec3(r, g, b);
}
//--------------------------------------------------------------------------------
Viewer::~Viewer(){
    // there is nithing to clean;
}
//--------------------------------------------------------------------------------
const glm::mat4 Viewer::calc_mvp(){
    auto rot_tmp = get_rot_tmp();
    if (! axis_sw){
        MVP = glm::translate(ort, glm::vec3(tr*scale,0.f))* glm::translate(glm::toMat4( rot_tmp * orient), -pos);
    } else {
        float m = (float)std::min(width, height);
        MVP = glm::ortho(-1.f*width/m, 1.f*width/m, -1.f*height/m, 1.f*height/m,
                1.f, -1.f) * glm::toMat4( rot_tmp * orient); // z also inverted by glm
    }
    return MVP;
}
const glm::mat4 Viewer::calc_itmvp() const{
    if (!axis_sw){
        return glm::inverseTranspose(MVP);
    }
    else return glm::mat4(1.0);// it's an identity matrix if that is not clear
}
//--------------------------------------------------------------------------------
void Viewer::set_view(float pitch, float yaw, float roll){
    orient = glm::quat(glm::vec3(pitch, yaw, roll))*glm::quat(glm::vec3(-M_PI_2,-M_PI_2,0.f));
}
void Viewer::get_view(float & pitch, float & yaw, float & roll) const{
    const glm::vec3 view = glm::eulerAngles(orient*glm::inverse(glm::quat(glm::vec3(-M_PI_2,-M_PI_2,0.f))));
    pitch =view.x;
    yaw = view.y;
    roll = view.z;
}
//--------------------------------------------------------------------------------
void Viewer::set_pos(float x, float y, float z){
    pos = glm::vec3(x,y,z);
}
////--------------------------------------------------------------------------------
void Viewer::get_pos(float* p){
    p[0] = pos.x;
    p[1] = pos.y;
    p[2] = pos.z;
}
//void Viewer::set_xrange(float lower, float upper){
//    min[0] = lower; max[0]= upper;
//}
////--------------------------------------------------------------------------------
//void Viewer::set_yrange(float lower, float upper){
//    min[1] = lower; max[1]= upper;
//}
////--------------------------------------------------------------------------------
//void Viewer::set_zrange(float lower, float upper){
//    min[2] = lower; max[2]= upper;
//}
//--------------------------------------------------------------------------------
void Viewer::display(){
    glClearColor(background.r, background.g, background.b, 0.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
//--------------------------------------------------------------------------------
void Viewer::plot(ShaderProg * spr){
    spr->AttachUniform(mvp_loc, "MVP");
    spr->AttachUniform(it_mvp_loc, "itMVP");
    spr->AttachUniform(vmin, "vmin");
    spr->AttachUniform(vmax, "vmax");
    spr->AttachUniform(vport, "viewport");
    spr->AttachUniform(unif_scale, "scale");
    if (mvp_loc != -1){
        const glm::mat4 & _MVP = calc_mvp();
        glUniformMatrix4fv( mvp_loc, 1, GL_FALSE, glm::value_ptr(_MVP));
    }
    if (it_mvp_loc != -1){
        const glm::mat4 & itMVP = calc_itmvp();
        glUniformMatrix4fv( it_mvp_loc, 1, GL_FALSE, glm::value_ptr(itMVP));
    }
    if (vmin != -1){
        auto va = get_vmin();
        glUniform3f(vmin, va.x,va.y,va.z);
    }
    if (vmax != -1){
        auto va = get_vmax();
        glUniform3f(vmax, va.x,va.y,va.z);
    }
    if (vport != -1){
        if (!axis_sw) glUniform2f(vport, width, height);
        else glUniform2f(vport, width/10., height/10.);
    }
    if (unif_scale != -1){
        if (!axis_sw) glUniform1f(unif_scale,scale);
        else glUniform1f(unif_scale,1.0);
    }
}

//--------------------------------------------------------------------------------
void Viewer::_reshape(int w, int h){
    glViewport(0, 0, w, h);
    width = w; height = h;
    min_size = std::min(w, h);
    int l = min_size;
    ort = glm::ortho(-w*scale/l, w*scale/l, -h*scale/l, h*scale/l,
                    scale, -scale); //z is inverted by glm
}
//--------------------------------------------------------------------------------
void Viewer::reshape(int w, int h){
    _reshape(w,h);
    glutPostRedisplay();
}
//--------------------------------------------------------------------------------
void Viewer::axis_switch(){
    axis_sw = !axis_sw;
    int sf = axis_sw?10:1; glViewport(0, 0, width/sf, height/sf);
}
//--------------------------------------------------------------------------------
float Viewer::get_scale() const{
    return scale;
}
glm::vec3 Viewer::get_vmin() const{
    if (axis_sw) return glm::vec3(-1.f,-1.f,-1.f);
    return min;
}
glm::vec3 Viewer::get_vmax() const{
    if (axis_sw) return glm::vec3(1.f,1.f,1.f);
    return max;
}
float Viewer::get_bounding_box(int i, bool mm) const{
    const glm::vec3 & getting = mm ? max: min;
    int j =0; float rej_value= getting.x;
    for(const float & c: {getting.x, getting.y, getting.z}){
        if(i==j) return c;
        j++;
        rej_value = mm? std::max(rej_value, c): std::min(rej_value, c);
    }
    return rej_value;
}
void Viewer::set_bounding_box(int i, float v,  bool mm){
    glm::vec3 & changing = mm ? max: min;
    int j =0;
    for(float * c: {&changing.x, &changing.y, &changing.z}){
        if(i==j) {*c = v; break;}
        j++;
    }
}
//--------------------------------------------------------------------------------
void Viewer::set_scale(float sc){
    scale = sc;
}
//--------------------------------------------------------------------------------
//static std::string Viewer::read_string(){
//};
std::string read_string(){
    std::string input_command;
    std::cin>>input_command;
    return input_command;
};
//--------------------------------------------------------------------------------
std::string Viewer::get_command(){
    std::string com;
    if (!command_fut.valid()) {
        command_fut = std::async(std::launch::async, read_string);
        com = std::string("");
    }
    std::chrono::milliseconds span (1);
    bool done = (command_fut.wait_for(span) == std::future_status::ready);
    if (done) com = command_fut.get();
    return com;
}
//--------------------------------------------------------------------------------
void Viewer::automove(){
    glm::vec3 cent = (max + min)*0.5f;
    scale = glm::distance(cent, max);
    pos = cent;
    if (scale == 0.f) scale = 1.f;
    //std::cout<<scale<<" "<<glm::to_string(pos)<<std::endl;
    _reshape(width, height);
}
////template void Viewer::autoscale<1>(Surface<1> * Sur);
////template void Viewer::autoscale<2>(Surface<2> * Sur);
//template void Viewer::autoscale<float>(SurfTemplate<float> * Sur);
