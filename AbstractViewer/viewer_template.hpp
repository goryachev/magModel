#ifndef VIEWER_TEMPLATE
#define VIEWER_TEMPLATE
#define GLM_FORCE_RADIANS
#ifndef DEBIAN
#define Sens (1.f)
#else //DEBIAN
#define Sens ((float)(180.f/M_PI))
#endif //DEBIAN
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
//#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/normal.hpp>
//#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include <aivlib/mystream.hpp>
#include <string>
#include <future>
#include "shaderprog.hpp"
class ShaderProg;
//#define sur_size 4
// добавить обертку для шейдеров
// это поможет сразу аттачить аттрибуты
// а не думать какому что нужно.
// Нужно проработать этот момент
// вероятно придестя использовать абстрактный класс.
// Шейдеры должны связываться с моделью, но можно их и развести, если сразу
// разобрать несколько вариантов.
// модели тоже абстрактным классом выдавать, вероятно упростит дело
// пока захардкодим, потом рефакторить будем)
// Выделить в отдлеьную функцию определение scale из размера коробки
void checkOpenGLerror();
const float pal[12] = {
    0.f, 0.f, 0.f,
    1.f, 0.f, 0.f,
    0.f, 1.f, 0.f,
    0.f, 0.f, 1.f
};
struct vertex{
    GLfloat x,y,z;
};
//--------------------------------------------------------------------------------
//VIEWER
//--------------------------------------------------------------------------------
class Viewer{
    private:
        float rotx, roty;
        float  ox, oy;
        float scale;
        int width, height, min_size;
        int Ntr;
        glm::vec2 tr, tr0;
            glm::vec3 pos, min, max;
        bool left_click, right_click, wire, axis_sw;
        //GLint vattr, nattr, cattr, mvp_loc,
        //      unif_minmax;
        glm::quat orient;
        glm::mat4 MVP,ort;
        std::future<std::string> command_fut;
        GLint mvp_loc, it_mvp_loc, vmin, vmax, vport, unif_scale;
        glm::vec3 background;
        void _reshape(int w, int h);
        //std::string com;
//#ifndef PYTHON
        //Texture * tex;
        //Surface<2> *sur;
        //std::ifstream * inp;
//#endif //PYTHON
        glm::quat get_rot_tmp() const;
    public:
        //ShaderProg * spr;
        //void rescale(float mult);
        Viewer();
        ~Viewer();
        //--------------------------------------------------------------------------------
        // functions for user
        void get_background(float & r, float & g, float & b);
        void set_background(float r, float g, float b);
        float get_scale() const;
        void axis_switch();
        void get_pos(float* p);
        glm::vec3 get_vmin() const;
        glm::vec3 get_vmax() const;
        int get_width() const {return width;}
        int get_height() const {return height;}
        bool get_wire() const {return wire;}
        void set_view(float pitch, float yaw, float roll);
        void get_view(float & pitch, float & yaw, float & roll)const;
        void set_pos(float x, float y, float z);
        void set_scale(float sc);
        void set_wire(bool tf);
        float get_bounding_box(int i, bool mm)const;
        void set_bounding_box(int i, float v ,bool mm);
        void automove();
        //void set_xrange(float lower, float upper);
        //void set_yrange(float lower, float upper);
        //void set_zrange(float lower, float upper);
        std::string get_command();
        //void rotate(  float x, float y, float z);
        const glm::mat4 calc_mvp();
        const glm::mat4 calc_itmvp() const;
        //void set_view(float x, float y, float z);
        //void set_pal();
        //--------------------------------------------------------------------------------
        // биндинги будут в python
        // functions for GUI
        //template<class T> void autoscale(SurfTemplate<T> * Sur);
        //template<class T> void setminmax(SurfTemplate<T> * Sur);
        void drag(int x,int y);
        void mouse_click(int button, int state, int x, int y);
        void clip_plane_move(float shift, int num);
        void plot(ShaderProg * spr);
        void display(void);
        void reshape(int w, int h);
        void togglewire();
        //void rotatexy(int x, int y);
        //void special(int key, i)
        //--------------------------------------------------------------------------------
        //void shader_init();
        void GL_init();

};
#endif //VIEWER_TEMPLATE
