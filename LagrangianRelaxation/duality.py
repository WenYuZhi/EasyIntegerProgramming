import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt

class Duality:
    def __init__(self, u_init) -> None:
        self.u = u_init
        self.obj_value_trace, self.grad_norm_trace = [], []
        self.step_trace, self.gap_trace = [], []
    
    def get_subgrad(self, relaxed_model):
        self.relaxed_model = relaxed_model
        self.subgrad = np.array([self.relaxed_model.subgrad[i]() for i in self.relaxed_model.N])
        return self.subgrad
    
    def get_step_size(self):
        self.best_bound = 3552.0
        self.grad_norm = np.linalg.norm(self.subgrad)
        self.step_size = (self.best_bound - value(self.relaxed_model.obj)) / (self.grad_norm)**2
        self.__trace_results()
        
    def __trace_results(self):
        self.obj_value_trace.append(value(self.relaxed_model.obj))
        self.step_trace.append(self.step_size)
        self.grad_norm_trace.append(self.grad_norm)
        self.gap_trace.append(self.__get_gap())
    
    def __get_gap(self):
        self.gap = 100*(self.best_bound - value(self.relaxed_model.obj)) / self.best_bound
        return self.gap

    def update(self):
        self.u += self.step_size * self.subgrad
        return self.u
    
    def print_status(self, k, max_iter):
        self.flag = max_iter // 20
        if k // self.flag == 0 or k == max_iter - 1:
            print("obj value {:0.2f}  ".format(value(self.relaxed_model.obj)), end='')
            print("step size {:0.4f}  ".format(self.step_size), end='')
            print("gap {:0.4f}(%)  ".format(self.gap))
    
    def plot_trace(self):
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.obj_value)
        plt.ylabel('relaxed objective function')
        plt.xlabel('iteration times')
        plt.subplot(2,2,2)
        plt.plot(self.grad_norm_trace)
        plt.ylabel('subgrad norm')
        plt.xlabel('iteration times')
        plt.subplot(2,2,3)
        plt.plot(self.step_trace)
        plt.ylabel('step size')
        plt.xlabel('iteration times')
        plt.subplot(2,2,4)
        plt.plot(self.gap)
        plt.ylabel('gap(\%)')
        plt.xlabel('iteration times')
        plt.tight_layout()
        plt.show()