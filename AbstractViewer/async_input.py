#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import multiprocessing
import Queue
import sys
import time
#import atexit
import readline
import rlcompleter
import signal
import os
class KillException(Exception):
    pass
def async_process(f):
# here we have a problem with thread stop
#solution queue + timeout
    #def wrapped_f(q,l, *args, **kwargs):
    #    '''this function calls the decorated function and puts the
    #    result in a queue'''
    #    l.acquire()
    #    ret = f(*args, **kwargs)
    #    l.release()
    #    q.put(ret)

    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached'''
        t = multiprocessing.Process(target=f, args=args, kwargs=kwargs)
        t.start()
        return t
    return wrap

class rl_async_reader:
    def __init__(self, stdin_fid,history_path,prompt = ">"):
        self.stdin_fid = stdin_fid
        self.history_path = history_path
        self.prompt = prompt
        readline.parse_and_bind('tab: complete')
        self.queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.KillException = KillException
    def set_completer(self, namespace):
        self.rlc = rlcompleter.Completer(namespace)
        readline.set_completer(self.rlc.complete)
    def read_history(self):
        if os.path.exists(self.history_path):
            readline.read_history_file(self.history_path)
    def write_history(self):
        readline.write_history_file(self.history_path)
    def kill_handler(self,signum, frame):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise self.KillException
    def pre_input_hook(self):
        signal.signal(signal.SIGHUP, self.kill_handler)
    def get(self):
        try:
            return self.queue.get_nowait()
        except (Queue.Empty):
            return ""
    #@threaded
    @async_process
    def __call__(self):
        import sys,os,time,signal
        sys.stdin = os.fdopen(self.stdin_fid)
        import readline
        readline.set_pre_input_hook(self.pre_input_hook)
        readline.set_completer(self.rlc.complete)
        self.read_history()
        try:
            self.lock.acquire()
            while(1):
                self.queue.put(raw_input(self.prompt))
        except (self.KillException, EOFError, KeyboardInterrupt):
            self.queue.put("exit()")
            self.lock.release()
            self.write_history()
