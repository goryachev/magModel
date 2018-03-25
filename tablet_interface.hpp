#ifndef TABLET_INTERFACE
#define TABLET_INTERFACE
#include "AbstractViewer/plottable.hpp"
#include "tablet.hpp"
#include <aivlib/vectorD.hpp>
class TabletInterface:public TabletIF, public Plottable{
	private:
        std::vector<unsigned int> select;
        GLint unif_radius;
        //cudaGraphicsResource * res_mom, * res_pos;
        void * res_mom;
        void * res_pos;
    public:
        float radius;
        ~TabletInterface();
        TabletInterface();
        void map_resources();
        void unmap_resources();
        void load_on_device();
        void reload_select();
        void set_select(int * ids, int len);
        void select_all();
        void get_auto_box(aiv::vctr<3, float> & bb_min, aiv::vctr<3, float> & bb_max);
        void attach_shader(ShaderProg * spr); 
        void plot(ShaderProg * spr);
};
#endif // TABLET_INTERFACE

