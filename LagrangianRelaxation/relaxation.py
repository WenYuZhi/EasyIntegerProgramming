import numpy as np
from pyomo.environ import *

class Relaxation:
    def __init__(self) -> None:
        pass
    
    def relax_constrs(self, relaxed_constrs):
        relaxed_constrs.deactivate()

    def set_objective(self, model, relaxed_obj):
        model.obj = Objective(rule = relaxed_obj, sense = minimize)