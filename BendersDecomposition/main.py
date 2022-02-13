from grpc import Status
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
from sympy import N

class Subproblem:
    def __init__(self, N, M) -> None:
        self.N, self.M = N, M
        self.m = gp.Model("subproblem")
        self.u = self.m.addVars(N+M, lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'u')

    def add_constrs(self, A, c):
        self.m.addConstrs((gp.quicksum(A[j,i]*self.u[j] for j in range(A.shape[0])) >= c[i]) for i in range(A.shape[1]))
    
    def set_objective(self, B, b, y):
        self.p = (b - np.dot(B, y)).reshape(self.N + self.M)
        self.m.setObjective(gp.quicksum(self.p[i]*self.u[i] for i in range(self.N + self.M)), sense = GRB.MINIMIZE)
    
    def solve(self):
        self.m.Params.InfUnbdInfo = 1
        self.m.optimize()
    
    def get_status(self):
        if self.m.Status == GRB.Status.UNBOUNDED or GRB.Status.INF_OR_UNBD:
            return np.array([x.getAttr('UnbdRay') for x in self.m.getVars()])
            # return [x.LB for x in self.m.getVars()]
        elif self.m.Status == GRB.Status.OPTIMAL:
            return self.get_solution()
        else:
            return None
    
    def get_solution(self):
        return np.array([self.m.getVars()[i].x for i in range(self.M + self.N)])
    
    def write(self):
        self.m.write("sub_model.lp")

class Master:
    def __init__(self, N, M, d) -> None:
        self.N, self.M, self.d = M, N, d
        self.m = gp.Model("Master")
        self.y = self.m.addVar(lb = 0.0, ub = GRB.INFINITY, vtype = GRB.INTEGER, name = 'y')
        self.z = self.m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'z')
    
    def add_cut1(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.p[0]*self.y <= self.q[0])
    
    def add_cut2(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.p[0]*self.y + self.z <= self.q[0])
    
    def set_objective(self):
        self.m.setObjective(self.d*self.y + self.z, sense = GRB.MAXIMIZE)
    
    def solve(self):
        self.m.optimize()

    def get_solution(self):
        return self.m.getVars()[0].x 
    
    def write(self):
        self.m.write("model.lp")


N, M = 10, 1
c, d = np.array([1+0.01*i for i in range(N)]), 1.045
A, B = np.vstack((np.ones((1, N)), np.eye(N))), np.array([1 if i == 0 else 0 for i in range(N+1)]).reshape(N+1,1)
b = np.array([1000 if i == 0 else 100 for i in range(N+1)]).reshape(N+1,1)

ub, lb = np.inf, -np.inf
MAX_ITER_TIMES, eps = 1, 1.0

subproblem = Subproblem(N, M)
subproblem.add_constrs(A, c)
masterproblem = Master(N, M, d)
masterproblem.set_objective()
y = 1500

for i in range(MAX_ITER_TIMES):
    if ub - lb <= eps:
        break
    subproblem.set_objective(B, b, y = y)
    subproblem.solve()
    subproblem.write()
    rays = subproblem.get_status()

    masterproblem.add_cut1(B, b, u = rays)
    masterproblem.solve()
    masterproblem.write()
    y = masterproblem.get_solution()
    ub = masterproblem.m.ObjVal
    print("y: {}".format(y))
    
