#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# we assume that all libs imported in advance, since they all building in one file
# in addition we assume that all neded objects are in defaul namespace
from OpenGL.GLUT import *
import os.path
from PIL import Image
from OpenGL import GL
from aivlib.vctf3 import *
#from collections import MutableMapping
import time
import signal
import multiprocessing
import threading
from async_input import *
from viewer import *
from string import Formatter
from math import *

def threaded(f):
    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached'''
        t = threading.Thread(target=f, args=args, kwargs=kwargs)
        t.start()
        return t
    return wrap

KeysList = " `1234567890qwertyuiop[]asdfghjkl;'xcvbnm,,./~!@#$%^&*()_-+QWERTYUIOP{}ASDFGHJKL:\"|\\ZXCVBNM<>?"
SpecialKeysList = [GLUT_KEY_F1, GLUT_KEY_F2, GLUT_KEY_F3,
        GLUT_KEY_F4, GLUT_KEY_F5, GLUT_KEY_F6, GLUT_KEY_F7,
        GLUT_KEY_F8, GLUT_KEY_F9, GLUT_KEY_F10, GLUT_KEY_F11,
        GLUT_KEY_F12, GLUT_KEY_LEFT, GLUT_KEY_UP, GLUT_KEY_RIGHT,
        GLUT_KEY_DOWN, GLUT_KEY_PAGE_UP, GLUT_KEY_PAGE_DOWN,
        GLUT_KEY_HOME, GLUT_KEY_END, GLUT_KEY_INSERT]
def get_arglist(func, ommit_self = True):
    arglist = func.func_code.co_varnames[:func.func_code.co_argcount]
    if ommit_self and arglist and arglist[0]=="self": return arglist[1:]
    else: return arglist
def func2string(func):
    arglist = get_arglist(func)
    argum =(" {},"*(len(arglist)-1) + ' {} ') if arglist else ""
    return func.func_name + '('+argum.format(*arglist)+')'
DefaultKeyMapping = [ ("next()", " "), ("jump(-1)"," ", ["Shift"]),
        ("keyhelp()","h",["Shift"]), ("exit()", "q",["Shift"]),("keyhelp()","h"), ("exit()", "q"),
        ("autoscale()","A"), ("autoscale()","A",["Shift"]),
        ("saveimage(get_image_name())","s"), ("saveimage(get_image_name())","s",[ "Shift" ]),
        ("toggle_wire()","w"), ("toggle_wire()","w",["Shift"]),
        ("ax^=True","c"), ("ax^=True","c",["Shift"]),
        ("rotate(-10,0)", GLUT_KEY_LEFT), ("rotate(10,0)", GLUT_KEY_RIGHT),
        ("rotate(0,-10)", GLUT_KEY_UP), ("rotate(0,10)", GLUT_KEY_DOWN),
        ("zoom(10)", GLUT_KEY_UP,["Ctrl"]), ("zoom(-10)", GLUT_KEY_DOWN, ["Ctrl"]),
        ("move(0,-10)", GLUT_KEY_UP,["Shift"]), ("move(0,10)", GLUT_KEY_DOWN, ["Shift"]),
        ("move(10,0)", GLUT_KEY_RIGHT,["Shift"]), ("move(-10,0)", GLUT_KEY_LEFT, ["Shift"]),
        ("clip_plane_move(0.01,0)", GLUT_KEY_RIGHT,[],["1"]), ("clip_plane_move(-0.01,0)", GLUT_KEY_LEFT, [],["1"]),
        ("clip_plane_move(0.01,1)", GLUT_KEY_RIGHT,[],["2"]), ("clip_plane_move(-0.01,1)", GLUT_KEY_LEFT, [],["2"]),
        ("clip_plane_move(0.01,2)", GLUT_KEY_RIGHT,[],["3"]), ("clip_plane_move(-0.01,2)", GLUT_KEY_LEFT, [],["3"]),
        ("clip_plane_move(0.01,3)", GLUT_KEY_RIGHT,[],["4"]), ("clip_plane_move(-0.01,3)", GLUT_KEY_LEFT, [],["4"]),
        ("clip_plane_move(0.01,4)", GLUT_KEY_RIGHT,[],["5"]), ("clip_plane_move(-0.01,4)", GLUT_KEY_LEFT, [],["5"]),
        ("clip_plane_move(0.01,5)", GLUT_KEY_RIGHT,[],["6"]), ("clip_plane_move(-0.01,5)", GLUT_KEY_LEFT, [],["6"]) ]
# add cutting surfaces
SurfTemplateKeys =  [ ("extendrange(1.1)","+",["Ctrl"]), ("extendrange(1.1)","+",["Ctrl","Shift"]),
        ("extendrange(1/1.1)","-",["Ctrl"]), ("extendrange(1/1.1)","-",["Ctrl","Shift"]),
        ("rangemove(1/1.1-1.0,True)","{",["Shift"]), ("rangemove(1/1.1-1.0,True)","{"),
        ("rangemove(1/1.1-1.0,False)","}",["Shift"]), ("rangemove(1/1.1-1.0,False)","}"),
        ("rangemove(0.1,True)","[",["Shift"]), ("rangemove(0.1,True)","["),
        ("rangemove(0.1,False)","]",["Shift"]), ("rangemove(0.1,False)","]") ]

class UniversalViewer:
    def __init__(self, argv):
        glutInitContextVersion(3,3)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInit(argv)
        self.V = Viewer()
        glutInitWindowSize(self.V.get_width(), self.V.get_height())
        glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH)
        glutCreateWindow("viewer")
        self.V.GL_init()
        self.spr = ShaderProg()
        checkOpenGLerror()
        self.palettes = {}
        self.add_pal("pal", [1.,0.,0., 1.,.5,0., 1.,1.,0., 0.,1.,0., 0.,1.,1., 0.,0.,1., 1.,0.,1.])
        self.add_pal("rgb", [1.,0.,0.,0.,1.,0.,0.,0.,1.])
        checkOpenGLerror()
        self.Axis = Axis()
        checkOpenGLerror()
        self.rl_reader = rl_async_reader(sys.stdin.fileno(), os.path.expanduser("~/.UniversalViewer"))
        self.Axis.load_on_device()
        checkOpenGLerror()
        self.savebuffer = FrameBuffer(self.V.get_width(), self.V.get_height());
        checkOpenGLerror()
        d ={}
        self._closed = False
        for k in SpecialKeysList:
            d[k] = []
        self.SpecialKeyUp = dict([ (k, []) for k in SpecialKeysList])
        self.SpecialKeyDown = dict([ (k, []) for k in SpecialKeysList])
        self.KeyUp = dict([(k, []) for k in KeysList])
        self.KeyDown = dict([(k, []) for k in KeysList])
        self.keystates = dict([(k, False) for k in KeysList])
        self.ax = True
        self.bb_auto = True
        self.idle_actions = []
        self.title_template = "UniversalViewer scale:{scale}"
        self.image_name_template = "SC{scale}-V{view[0]}_{view[1]}_{view[2]}.png"
        self.set_pal("pal")
        self.__help = threading.RLock()
        self.buffers = {}
        self.buffer = None
        self.namespace = {}
#--------------------------------------------------------------
    def __setattr__(self,name,value):
        self.__dict__[name] = value
    def __setitem__(self, key, value):
        if "set_"+key in self:
            if isinstance(value, tuple) or isinstance(value, list):
                getattr(self,"set_"+key)(*value)
            elif isinstance(value, dict):
                getattr(self,"set_"+key)(**value)
            else: getattr(self,"set_"+key)(value)
        else:
            self.__dict__[key] = value
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            try:
                return getattr(self,"get_"+key)()
            except AttributeError:
                raise KeyError("get_"+key)
    def __contains__(self, key):
        k_in_d = key in self.__dict__ or key in self.__class__.__dict__
        gsk_in_d = "set_"+key in self.__dict__ and "get_"+key in self.__dict__
        gsk_in_cd = "set_"+key in self.__class__.__dict__ and "get_"+key in self.__class__.__dict__
        return k_in_d or gsk_in_d or gsk_in_cd
    def execute(self, string, **kwargs):
        "выполняет функции в переменных вьюера, служебная функциия"
        kwargs.update(globals())
        if isinstance(string, str): exec(string,kwargs,self)
        else: exec(func2string(string), kwargs, self)
        glutPostRedisplay()
    def evaluate(self, mystring, **kwargs):
        "вычисляет выражение в переменных вьюера, служебная функциия"
        kwargs.update(globals())
        print mystring
        if isinstance(mystring, str): return eval(mystring,kwargs,self)
        else: return eval(func2string(mystring), kwargs, self)
    def set_key(self, func, key, modificators = [], key_pressed = [], up = False):
        '''Установливает сочетание клавиш, key — клавиша к которой пивязвнв функция, func — функция для выполнения в виде строки,
        modificators — список нажатых специальных клавиш "Ctrl", "Shift", "Alt" по дефолту пустой,
        key_pressed — спислк клавиш, которые должны быть зажаты, key не считается, up — булево число, True если действие выполняется
        на отпускание, False — усли на нажатии, по умолчанию False'''
        if(( key not in list(KeysList))  and (key not in SpecialKeysList)): print "Unhandled key {}".format(key); return
        KeysDict = (self.KeyDown, self.KeyUp, self.SpecialKeyDown, self.SpecialKeyUp)[ int(key in SpecialKeysList) *2 + int(up)]
        KeysDict[key].append((func, modificators, key_pressed))
#-------------------------------------------------------------
    def states(self):
        "Возвращает состояние клавиш модификаторов в словаре"
        state = glutGetModifiers()
        return {"Shift":bool(state & 0x1), "Ctrl":bool(state & 0x2), "Alt":bool(state & 0x4)}
    def add_pal(self, name, pal_list):
        '''Добавляет палитру с именем name и цветами заданными в виде списка float со значениями от 0 до 1,
        они групируются по 3 формируя цвета, длина округляется до ближайшей снизу кратной 3'''
        truncate3 = lambda x: x - x%3
        nlen = truncate3(len(pal_list))
        pal = float_array(nlen)
        for i, v in enumerate(pal_list[:nlen]): pal[i] = v
        self.palettes[name] = Texture(pal, nlen/3)
    def eval_template(self, template):
        "Вычисляет значение строки-шаблона"
        return Formatter().vformat(template,[],self)
    def get_image_name(self):
        "Вычисляет значение изображения по умолчанию, используя image_name_template"
        return self.eval_template(self.image_name_template)
    def get_title(self):
        "Вычисляет значение заголовка, используя titel_template"
        return self.eval_template(self.title_template)
    def get_key_function(self, KeyDict):
        "Возвращает фунеции для клавиш, служебная функция"
        def KeyHandler(k, x, y):
            #print k
            if KeyDict == self.KeyDown :
                self.keystates[k] = True
            elif KeyDict == self.KeyUp:
                self.keystates[k] = False
                self.keystates[k.upper()] = False
            if k not in KeyDict or not KeyDict[k]: return
            mod = self.states()
            for func, modifiers, key_pressed in KeyDict[k]:
                execute = True
                #print func, k
                for m in ["Ctrl", "Shift", "Alt"]:
                    execute &= (mod[m] == (m in modifiers))
                    #print mod[m] == (m in modifiers)
                    if not execute: break
                #print execute, modifiers, mod
                for kp in KeysList:
                    if (kp == k): continue
                    execute &= (self.keystates[kp] == (kp in key_pressed))
                    if not execute: break
                #print execute, key_pressed
                if execute:
                    #func_run(func, x=x, y = y, **self.params)
                    self.execute(func, x=x,y=y)
                    break
            glutPostRedisplay()
        return KeyHandler
    def help(self, myobject=None):
        "Показать справку на обьект"
        @threaded
        def help_thread(self, myobject):
            self.__help.acquire()
            os.kill(self._t.pid,signal.SIGSTOP)
            self.rl_reader.lock.release() # Мы его остановили так что можно освободить
            self.rl_reader.lock.acquire() # и занять lock на чтенеие
            if myobject is None:
                help(self)
            else: help(myobject)
            # освобождение замка на чтение тут не нужно, т.к. продолжение
            # _t его должно захватывать
            os.kill(self._t.pid,signal.SIGCONT)
            self.__help.release()
        help_thread(self,myobject)
    def set_pal(self, pal_name):
        "Устанавливает палитру"
        self.tex = self.palettes[pal_name]
        self.cur_pal = pal_name
    def set_view(self,pitch, yaw, roll):
        "Устанавливает вид по углам Эйлера порядок y z y"
        self.V.set_view(pitch,yaw,roll)
    def set_pos(self, x, y, z):
        "Устанавливает смещение"
        self.V.set_pos(x,y,z)
    def set_scale(self, sc):
        "Устанавливает масштаб"
        self.V.set_scale(sc)
    def set_xrange__(self, lower, upper):
        "Устанавливает диапазон значений по x, не меняет bb_auto"
        #self.bb_auto = False
        self.V.set_bounding_box(0, lower, False), self.V.set_bounding_box(0, upper, True)
    def set_yrange__(self, lower, upper):
        "Устанавливает диапазон значений по y, не меняет bb_auto"
        #self.bb_auto = False
        self.V.set_bounding_box(1, lower, False), self.V.set_bounding_box(1, upper, True)
    def set_zrange__(self, lower, upper):
        "Устанавливает диапазон значений по z, не меняет bb_auto"
        #self.bb_auto = False
        self.V.set_bounding_box(2, lower, False), self.V.set_bounding_box(2, upper, True)
    def set_xrange(self, lower, upper):
        "Устанавливает диапазон значений по x, меняет bb_autp на False "
        self.bb_auto = False
        self.set_xrange__(lower,upper)
    def set_yrange(self, lower, upper):
        "Устанавливает диапазон значений по y, меняет bb_autp на False "
        self.bb_auto = False
        self.set_yrange__(lower,upper)
    def set_zrange(self, lower, upper):
        "Устанавливает диапазон значений по z, меняет bb_autp на False "
        self.bb_auto = False
        self.set_zrange__(lower,upper)
    def get_scale(self):
        "Вовращает текущий масштаб"
        return self.V.get_scale()
    def get_width(self):
        "Возвращает ширину окна"
        return self.V.get_width()
    def get_height(self):
        "Возвращает высоту окна"
        return self.V.get_height()
    def get_pal(self):
        "Возвращает имя текущей палитры"
        return self.cur_pal
    def get_xrange(self):
        "Возвращает диапазон значений x"
        return self.V.get_bounding_box(0, False), self.V.get_bounding_box(0, True)
    def get_yrange(self):
        "Возвращает диапазон значений y"
        return self.V.get_bounding_box(1, False), self.V.get_bounding_box(1, True)
    def get_zrange(self):
        "Возвращает диапазон значений z"
        return self.V.get_bounding_box(2, False), self.V.get_bounding_box(2, True)
    def get_pos(self):
        "Возвращает текущую позицию"
        p = float_array(3)
        self.V.get_pos(p)
        return (p[0], p[1], p[2])
    def get_view(self):
        "Возвращает углы Эулера для текущего вида"
        p,y,r = float_array(1),float_array(1),float_array(1)
        self.V.get_view(p,y,r)
        return (p[0],y[0],r[0])
    def toggle_wire(self):
        "Переключает между проволочной моделью и обычной"
        self.V.togglewire()
    def set_wire(self, b_tf):
        "Переключение между проволочной моделью и обычной, True — проволочная, False — обычная"
        self.V.set_wire(b_tf)
    def get_wire(self):
        "Возвращает True, если используется проволочная модель"
        return self.V.get_wire()
    def toggle_axis(self):
        "Переключает отображение осей"
        self.ax^=1
    def __mouse_click(self, but, ud, x,y):
        "Функция для мышки, служебная функция"
        if but in [GLUT_LEFT_BUTTON, 3,4] and ud == GLUT_DOWN:
            self.bb_auto=False
        self.V.mouse_click(but, ud, x, y)
    def rotate(self, x, y):
        "Поворот, в координатах окна"
        x_min = 0 if x>=0 else -x
        y_min = 0 if y>=0 else -y
        self.V.mouse_click(GLUT_RIGHT_BUTTON, GLUT_DOWN, x_min,y_min )
        self.V.drag(x+x_min,y)
        self.V.mouse_click(GLUT_RIGHT_BUTTON, GLUT_UP, x+x_min,y_min)
    def move(self, x, y):
        "Перемещение, в координатах окна"
        self.bb_auto=False
        x_min = 0 if x>=0 else -x
        y_min = 0 if y>=0 else -y
        self.V.mouse_click(GLUT_LEFT_BUTTON,GLUT_DOWN,x_min,y_min)
        self.V.drag(x,y)
        self.V.mouse_click(GLUT_LEFT_BUTTON,GLUT_UP,x,y)
    def zoom(self, x):
        "Зум, в координатах окна, положительное приблидает, отрицательное отдаляет, величина игнорируется"
        self.bb_auto=False
        if x<0: self.V.mouse_click(4,GLUT_DOWN,0,0)
        else: self.V.mouse_click(3,GLUT_DOWN,0,0)
    def idle(self):
        "Функция idle для окна"
        command = self.rl_reader.get()
        while(command):
            try:
                if isinstance (command, str):
                    self.execute(command)
                else: raise command
            except (NameError, SyntaxError, TypeError):
                import traceback, sys
                traceback.print_exception( *sys.exc_info())
                command = self.rl_reader.get()
            except:
                command = ""
                import traceback, sys
                traceback.print_exception( *sys.exc_info())
                self.exit()
            else: command = self.rl_reader.get()
        cur_time = time.time()
        for i, (name, last_time, interval, action) in enumerate(self.idle_actions):
            if cur_time-last_time>=interval:
                self.execute(action)
                self.idle_actions[i][1] = time.time()
    def add_idle_action(self, name, interval,action):
        "Добавляет действие с именем name в цикл idle, с интервалом в секундах interval и действием action в виде строки"
        for i,act in enumerate(self.idle_actions):
            if name ==act[0]:
                self.idle_actions[i]=[name,time.time(), interval,action]
                break
        else:
            self.idle_actions.append([name,time.time(), interval,action])
    def del_idle_action(self,name):
        "Удаляет действие с именем name из цикла idle"
        for i, (aname, lt, interv, action) in enumerate(self.idle_actions):
            if name == aname:
                del self.idle_actions[i]
    def clear_idle_actions(self):
        "Очищает цикл idle"
        del self.idle_actions
        self.idle_actions = []
    def plot(self,surf):
        "Устанавливает данне для отображения"
        self.Surf = surf
        self.Surf.load_on_device()
    #def subplot(*args);
    #    self.Surfaces += args
    def dump_params(self, **kwargs):
        "Сохраняет параметры по словарю"
        pars = {}
        for name,set_func in kwargs.items():
            if name == set_func:
                pars[name] = (self[name],set_func)
            else:
                result = self[name]()
                if not isinstance(result,list) and not isinstance(result, tuple):
                    result = (result,)
                pars[name] = (result, set_func)
        return pars
    def load_params(self, **params):
        "Загружает параметры по словарю"
        for name, val_func in params.items():
            #print name, val_func
            if name == val_func[1]:
                self[name] = val_func[0]
            else: self[val_func[1]](*val_func[0])
    def clip_plane_move(self, val,pl):
        '''Перемещает плоскости отсечения, эквивалентно перемещению соответствующих границ диапазонов значений
        на величину val внутрь диапазона. pl 0,1 и 2 нижние границы по x,y и z соответственно, 4,5,6 — верхние'''
        self.bb_auto = False
        self.V.clip_plane_move(val,pl)
    def switch_buffer(self, name,  save = True, SaveDict={}):
        '''Переключает текущее отображение на созраненное с именем name, если save — True созраняет текущие параметры отображения
        под текущим именем, SaveDict — словарь созраняемых параметров'''
        if (self.buffer is not None) and save:
            print "Save", self.buffer
            self.buffers[self.buffer] = (self.dump_params(**self.buffers[self.buffer][1]), self.buffers[self.buffer][1])
        self.buffer  = name
        if name in self.buffers:
            print "Load", self.buffer
            self.load_params(**self.buffers[name][0])
        print "Save", self.buffer
        self.buffers[name] = ( self.dump_params(**SaveDict), SaveDict )
        glutPostRedisplay()
    def next_buffer(self, save=True):
        "Переключает на следующее отображение и сохраняет текущее если Save установлен"
        if not self.buffers: return
        bn = self.buffers.keys()
        bn.sort()
        bi = bn.index(self.buffer)
        bi = (bi +1) % len(bn)
        self.switch_buffer(bn[bi], save, self.buffers[bn[bi]][1])
    def prev_buffer(self, save=True):
        "Переключает на предыдущее отображение и сохраняет текущее если Save установлен"
        if not self.buffers: return
        bn = self.buffers.keys()
        bn.sort()
        bi = bn.index(self.buffer)
        bi = bi - 1
        self.switch_buffer(bn[bi], save,self.buffers[bn[bi]][1])
    def autoscale(self):
        "Установить пределы по x,y,z чтобы отображаемые данные полностью помещались в окно, устанавливает bb_auto."
        bb_min, bb_max = self.Surf.autobox()
        self.set_xrange(bb_min[0], bb_max[0])
        self.set_yrange(bb_min[1], bb_max[1])
        self.set_zrange(bb_min[2], bb_max[2])
        self.bb_auto = True
        self.V.automove()
    #def autoscalecb(self):
    #    #self.cb_auto = True
    #    self.Surf.auto_cbrange()
    def _display(self):
        "Прототип функции display для окна, служебная функция"
        if self.bb_auto:
            self.autoscale()
        self.V.display()
        self.Surf.display(self.V, self.spr, self.tex)
        if self.ax:
            self.V.axis_switch()
            GL.glDepthFunc(GL.GL_GREATER) # Overdraw)
            self.spr.render(self.Axis, self.V, self.palettes["rgb"])
            GL.glDepthFunc(GL.GL_LEQUAL)
            self.spr.render(self.Axis, self.V, self.palettes["rgb"])
            self.V.axis_switch()
    def display(self):
        "Функция display для окна, служебная функция"
        self._display()
        glutSetWindowTitle(self.get_title())#self.execute(self.title_template))
        glutSwapBuffers()
    def exit(self):
        "Закрывает окно и завершает програму"
        if (self._closed == True): return
        self._closed = True
        os.kill(self._t.pid,signal.SIGHUP)
        self.rl_reader.lock.acquire()
        self.rl_reader.lock.release()
        self._t.join()
        glutLeaveMainLoop()
    def keyhelp(self):
        "Выводит справку по привязкам клавиш"
#тут не хватает нормального форматирования для комбинаций клавиш
        for kd in [self.KeyDown, self.SpecialKeyDown, self.KeyUp, self.SpecialKeyUp]:
            for key, flist in kd.items():
                for func, modifiers, key_pressed in flist:
                    print key, '	',func, modifiers, key_pressed
    def shader_load(self, vertex_string, fragment_string):
        "Загружает шейдеры из строк"
        self.spr.extern_load(vertex_string, fragment_string)
    def saveimage(self,name,width=None, height=None):
        "Сохраняет отображаемое изображение под именем name"
        if width is None:
            width = self.get_width()
        if height is None:  height= self.get_height()
        #from OpenGL import GL
        #myfb = GL.glGenFramebuffers(1)
        if width != self.savebuffer.width() or height != self.savebuffer.height():
            self.savebuffer.resize(width,height)
        old_w,old_h = self.get_width(), self.get_height()
        if old_w!= width or old_h != height:
            self.V.reshape(width, height)
        checkOpenGLerror()
        self.savebuffer.bind_draw()
        #GL.glReadBuffer(GL.GL_BACK)
        #GL.glRenderBuffer(Gl.GL_BACK)
        #GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, myfb)
        checkOpenGLerror()
        self.display()
        checkOpenGLerror()
        self.savebuffer.bind_read()
        checkOpenGLerror()
        buffer = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        checkOpenGLerror()
        #time.sleep(0.5)
        image = Image.frombytes(mode="RGBA", size=(width, height), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        print "saved to "+name
        image.save(name)
        image.close()
        self.savebuffer.relax()
        #GL.glDeleteFramebuffers(myfb)
        self.V.reshape(old_w,old_h)
        del buffer, image
    def set_title(self,template):
        "Изменяет шаблон заголовка, Шаблон является строкой для оператора форматирования"
        self.title_template = template
    def __sigterm_handler(self,signum,frame):
        "Сохранять историю при внезапном закрытии терминала, служебная функция"
        self.exit()
    def __call__(self): # start main loop
        "Запускает mainloop"
        glutSetWindowTitle(self.get_title())#self.exec(self.title_template.format(self.params)))
        glutDisplayFunc(self.display)
        glutMotionFunc(self.V.drag)
        glutMouseFunc(self.__mouse_click)
        glutKeyboardFunc(self.get_key_function( self.KeyDown))
        glutKeyboardUpFunc(self.get_key_function( self.KeyUp))
        glutSpecialFunc(self.get_key_function( self.SpecialKeyDown))
        glutSpecialUpFunc(self.get_key_function( self.SpecialKeyUp))
        glutReshapeFunc(self.V.reshape)
        glutIdleFunc(self.idle)
        glutCloseFunc(self.exit)
        self._t=self.rl_reader()
        signal.signal(signal.SIGHUP,self.__sigterm_handler)
        glutMainLoop()

