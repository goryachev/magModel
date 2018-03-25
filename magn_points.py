#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from aivlib.vctf3 import *
import mview
import time
from math import *
def list2intarr(args):
    ia=viewer.int_array(len(args));
    for i, v in enumerate(args):
        ia[i] =v
    return ia
class MagPoints:
    def __init__(self):
        "Инициализирует данные с помощью потока mystream"
        #print "interface init"
        self.Tablet = mview.TabletInterface()
        #self.frames=[0,]
        self.load_on_device()
        self.init()
        self.lattice = -1
        self.select_all()
        self.Tablet.radius =1./4#sqrt(3)/4.
        #self.set_select(0)
        #self.item = 0
        #self.cb_auto = True
    def init(self):
        "Загружает кадр из потока"
        #print "pre load"
        #self.Tablet.init()
        self.current_frame= 0
        #print check
    def select_all(self):
        "Выбирает все атомы для отображения"
        self.lattice = -1
        self.Tablet.select_all()
#    def select_lattice(self, i):
#        "Выбирает атомы подрешетки i для отображения"
#        self.lattice = i
#        self.Tablet.select_lattice(i)
    def set_select(self, *args):
        "Выбирает атомы по номерам для отображения"
        l = len(args)
        ia = mview.int_array(l)
        for i, ind in enumerate(args):
            ia[i] = ind
        self.Tablet.set_select(ia, l)
    def next(self, Nt=1):
        ''' Переходит к следующему фрейму'''
        #pos = self.Inp.tell()
        #print pos
        self.Tablet.map_resources()
        self.Tablet.run(Nt)
        self.Tablet.unmap_resources()
        self.current_frame+=1
        #if self.load():
        #    self.current_frame += 1
        #    #print self.current_frame
        #    if (self.current_frame == len(self.frames)): self.frames.append(pos)
        #    self.MView.reload_normals()
        #else:
        #    self.set_frame(0)
    #def set_frame(self,pos):
    #    "Устанавлевает фрейм pos, если он уже был просмотрен"
    #    if pos>= len(self.frames) or pos < -len(self.frames): return
    #    if pos<0: pos += len(self.frames)
    #    self.Inp.seek(self.frames[pos])
    #    self.load()
    #    self.current_frame = pos
    #    self.load_on_device()
    #def jump(self, pos):
    #    "Перепрыгивает на pos фреймов вперед, если цель уже была просмотрена"
    #    self.set_frame(self.current_frame + pos)
    def load_on_device(self):
        "Привязывает позиции шариков к буфферу OpenGL"
        self.Tablet.load_on_device()
    #def get_time(self):# do i need this method here ? apparently yes
    #    "Возвращает время текущего фрейма"
    #    return self.MView.time
    def get_frame(self):
        "Возвращает номер текущего фрейма"
        return self.current_frame
    def autobox(self):
        "Возвращает размер коробки включающей все отображаемые атомы"
        bbmin,bbmax = vctf3(), vctf3()
        self.Tablet.get_auto_box(bbmin, bbmax)
        return bbmin, bbmax
    def display(self,V,spr,tex):
        "Рисует картинку по данным, служебная функция"
        spr.render(self.Tablet, V, tex)
    def get_radius(self):
        "Возвращает радиус атома при отображении"
        return self.Tablet.radius
    def set_radius(self, r):
        "Устанавливает радиус атома при отображении"
        self.Tablet.radius = r
def ext_generator(name,*args):
    "Функция для генерации внешних методов, реализация скрытого частичного наследования"
    def ext_func(self,*args):
        return getattr(self.Surf, name)(*args)
    ext_func.__name__ = name
    ext_func.__doc__=getattr(MagPoints, name).__doc__
    return ext_func

