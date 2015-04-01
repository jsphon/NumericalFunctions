import pyximport; pyximport.install()
import cython_funcs.hello as hello

hello.say_hello_to('jon')


