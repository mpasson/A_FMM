import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sub_sm as sub
from matplotlib.backends.backend_pdf import PdfPages

class creator:
    def __init__(self,x_list=[],y_list=[],eps_lists=[]):
        self.x_list=x_list
        self.y_list=y_list
        self.eps_lists=eps_lists

    def slow_general(self,eps_core,eps_lc,eps_uc,w,h,t,Z):
        self.x_list=np.linspace(-0.5*w,0.5*w,len(Z)+1)
        self.y_list=[-0.5,-0.5*h,-0.5*h+t,0.5*h]
        self.eps_lists=[[eps_uc,eps_lc,eps_core,eps_uc]]
        eps=[eps_uc,eps_core]
        for i in Z:
            self.eps_lists.append([eps_uc,eps_lc,eps_core,eps[i]])
        

    def slow_2D(self,eps_core,eps_c,w,Z):
        self.x_list=np.linspace(-0.5*w,0.5*w,len(Z)+1)
        self.y_list=[-0.5]
        self.eps_lists=[[eps_c]]
        eps=[eps_c,eps_core]
        for i in Z:
            self.eps_lists.append([eps[i]])


    def ridge(self,eps_core,eps_lc,eps_uc,w,h,t,y_offset=0.0,x_offset=0.0):
        self.x_list=[-0.5*w+x_offset,0.5*w+x_offset]
        self.y_list=[-0.5,-0.5*h,-0.5*h+y_offset,-0.5*h+t+y_offset,0.5*h]
        self.eps_lists=[[eps_uc,eps_lc,eps_lc,eps_core,eps_uc],[eps_uc,eps_lc,eps_core,eps_core,eps_core]]

    def ridge_double(self,eps_core,eps_lc,eps_uc,w1,w2,h,t1,t2,y_offset=0.0,x_offset=0.0):
        self.x_list=[-0.5*w2+x_offset,-0.5*w1+x_offset,0.5*w1+x_offset,0.5*w2+x_offset]
        self.y_list=[-0.5,-0.5*h,-0.5*h+y_offset,-0.5*h+t2+y_offset,-0.5*h+t1+y_offset,0.5*h]
        self.eps_lists=[[eps_uc,eps_lc,eps_lc,eps_core,eps_uc,eps_uc],[eps_uc,eps_lc,eps_lc,eps_core,eps_core,eps_uc],[eps_uc,eps_lc,eps_core,eps_core,eps_core,eps_core],[eps_uc,eps_lc,eps_lc,eps_core,eps_core,eps_uc]]

    def rect(self,eps_core,eps_clad,w,h,off_x=0.0,off_y=0.0):
        self.x_list=[-0.5*w+off_x,0.5*w+off_x]
        self.y_list=[-0.5*h+off_y,0.5*h+off_y]
        self.eps_lists=[[eps_clad,eps_clad],[eps_clad,eps_core]]

    def slab(self,eps_core,eps_lc,eps_uc,w,offset=0.0):
        self.x_list=[-0.5,-0.5*w+offset,0.5*w+offset]
        self.y_list=[0.5]
        self.eps_lists=[[eps_uc],[eps_lc],[eps_core]]

    def slab_y(self,eps_core,eps_lc,eps_uc,w):
        self.x_list=[0.5]
        self.y_list=[-0.5,-0.5*w,0.5*w]
        self.eps_lists=[[eps_uc,eps_lc,eps_core]]

    def x_stack(self,x_l,eps_l):
        self.y_list=[0.5]
        self.x_list=[-0.5]+x_l
        self.eps_lists=[[eps_l[-1]]]
        for eps in eps_l:
            self.eps_lists.append([eps])

    def hole(self,h,w,r,e_core,e_lc,e_up,e_fill):
        self.x_list=[-0.5*w,-r,r,0.5*w]
        self.y_list=[-0.5*h,0.5*h,0.5]
        self.eps_lists=[[e_lc,e_up,e_up],[e_lc,e_core,e_up],[e_lc,e_fill,e_up],[e_lc,e_core,e_up]]

    def circle(self,e_in,e_out,r,n):
        self.x_list=np.linspace(-r,r,n)
        self.y_list=np.linspace(-r,r,n)
        [X,Y]=np.meshgrid(self.x_list,self.y_list)
        #ind= np.sqrt((X-0.5*r/float(n))**2+(Y-0.5*r/float(n))**2)<r
        ind= np.sqrt(X**2+Y**2)<r
        #eps=np.array([e_out,e_in])
        self.eps_lists=e_out+ind*(e_in-e_out)
        

