%module viewer
%exception { try{ $action }catch( const char *e ){ PyErr_SetString( PyExc_RuntimeError, e ); return NULL; }catch(...){ return NULL; } }
%pythoncode %{def _setstate( self, state ):
    if not hasattr( self, 'this' ) : self.__init__()
    self.__C_setstate__(state)
def import_aivlib(tgt) :
    name = tgt.split('-')[0]
    try : exec('import aivlib.'+name)
    except ImportError, e : import os; os.system('make-aivlib %r'%tgt); exec('import aivlib.'+name)
    return eval('aivlib.'+name)
def _swig_setattr(self,class_type,name,value):
    if name in class_type.__swig_setmethods__ : value = getattr( self, name ).__class__(value)
    return _swig_setattr_nondynamic(self,class_type,name,value,0)
__makefile__='Makefile'
%}
%typemap(out) bool&   %{ $result = PyBool_FromLong    ( *$1 ); %}
%typemap(out) char&   %{ $result = PyInt_FromLong     ( *$1 ); %}
%typemap(out) short&  %{ $result = PyInt_FromLong     ( *$1 ); %}
%typemap(out) int&    %{ $result = PyInt_FromLong     ( *$1 ); %}
%typemap(out) long&   %{ $result = PyInt_FromLong     ( *$1 ); %}
%typemap(out) float&  %{ $result = PyFloat_FromDouble ( *$1 ); %}
%typemap(out) double& %{ $result = PyFloat_FromDouble ( *$1 ); %}
%include "std_string.i"
%feature("autodoc","1");
%inline %{namespace aiv{}; %}
%include "carrays.i";
%ignore data_header; %ignore base_data;
%{
#include <aivlib/indexD.hpp>
#include <aivlib/vectorD.hpp>
#include <aivlib/memT.hpp>
#include <aivlib/arrayTD.hpp>
#include <aivlib/meshTD.hpp>
#include <aivlib/lrCubeTD.hpp>
#include <aivlib/base_sphere.hpp>
#include <aivlib/sphereT.hpp>
%}
%include "aivlib/indexD.hpp"
%include "aivlib/vectorD.hpp"
%include "aivlib/memT.hpp"
%include "aivlib/arrayTD.hpp"
%include "aivlib/meshTD.hpp"
%include "aivlib/lrCubeTD.hpp"
%include "aivlib/base_sphere.hpp"
%include "aivlib/sphereT.hpp"
%{
#include "shaderprog.hpp"
#include "viewer_template.hpp"
#include "plottable.hpp"
%}
%include "shaderprog.hpp"
%include "viewer_template.hpp"
%include "plottable.hpp"
%pythoncode%{ from aivlib.mystream import *; %}
%array_class(float, float_array);
