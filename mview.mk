name=mview
headers=AbstractViewer/plottable.hpp tablet.hpp tablet_interface.hpp# tablet.hpp
modules=AbstractViewer/viewer_template.cpp AbstractViewer/shaderprog.cpp AbstractViewer/plottable.cpp
#objects=tablet.o # tablet_interface.o
#objects=
LINKOPT=-lpthread -lglut -lGL -lGLU -lGLEW -L./ -ltablet
iheader ='%include "carrays.i";'
ifinish ='%array_class(float, float_array);\
		  %array_class(int, int_array);'
include aivlib/Makefile
