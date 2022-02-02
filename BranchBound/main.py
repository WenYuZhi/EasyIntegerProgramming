import gurobipy as gp
from gurobipy import GRB

def heuristic_solve(problem):
    problem.Params.OutputFlag = 0
    problem.optimize()
    if problem.status == GRB.INFEASIBLE:
        return None, None
    return problem.ObjVal, problem.getVars()

def choice_node(condidate_node):
    node = condidate_node.pop(0)
    return node, condidate_node

class Node:
    def __init__(self, model, upper_bound, lower_bound, candidate_vars):
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        self.model = model
        self.candidate_vars = candidate_vars.copy()
        assert(upper_bound >= lower_bound), "upper bound is less than lower bound"

    def optimize(self, heuristic_solve):
        self.obj_values, self.solution = heuristic_solve(self.model)
        if self.obj_values == None:
            return "infeasible"
        return "feasible"
    
    def update_upper_bound(self):
        if self.upper_bound > self.obj_values:
            self.upper_bound = self.obj_values
            assert(self.lower_bound <= self.obj_values)
            assert(self.lower_bound <= self.upper_bound), "upper bound is less than lower bound"
    
    def update_lower_bound(self):
        self.lower_bound = self.obj_values
        assert(self.lower_bound <= self.obj_values)
        assert(self.lower_bound <= self.upper_bound), "upper bound is less than lower bound"
    
    def is_integer(self):
        for var in self.solution:
            if 0 < var.x and var.x < 1:
                return False
        return True
    
    def is_child_problem(self):
        if self.candidate_vars:
            return True
    
    def get_child_problem(self):
        self.child_left, self.child_right = self.model.copy(), self.model.copy()
        branch_index, self.condidate_child_vars = self.choice_branch(self.candidate_vars)
        self.child_left.addConstr(self.child_left.getVars()[branch_index] == 0)
        self.child_right.addConstr(self.child_right.getVars()[branch_index] == 1)
        node_left = Node(self.child_left, self.upper_bound, self.lower_bound, self.condidate_child_vars)
        node_right = Node(self.child_right, self.upper_bound, self.lower_bound, self.condidate_child_vars)
        return node_left, node_right
    
    def choice_branch(self, candidate_vars):
        self.condidate_child_vars = self.candidate_vars.copy()
        branch_index = self.condidate_child_vars.pop(0)
        return branch_index, self.condidate_child_vars
    
    def write(self):
        self.model.write("model.lp")
    

model = gp.Model("mip1")
x = model.addVars(10, name = 'x', vtype = GRB.BINARY)

model.setObjective(x[0] + x[1] + 2*x[2] + 2*x[8] + x[9], GRB.MAXIMIZE)
model.addConstr(x[0] + 2*x[1] + 3*x[2] + 5*x[3] + 3*x[4] <= 8, "c0")
model.addConstr(2*x[3] + 2*x[4] + 3*x[5] + 5*x[6] + 3*x[7] <= 10, "c1")
model.addConstr(x[7] + x[8] + 3*x[9] <= 4, "c2")
model.addConstr(2*x[0] + x[2] + 3*x[7] + 3*x[8] + 2*x[9] <= 8, "c3")
model.addConstr(x[7] + x[8] + 3*x[9] >= 1, "c4")
model.addConstr(2*x[4] + 2*x[5] + x[6] + 5*x[7] + 3*x[8] >= 4, "c5")
model.optimize()
model.write("model_integer.lp")

upper_bound, lower_bound = float('inf'), 0
model_relax = model.relax()
root_node = Node(model = model_relax, upper_bound = upper_bound, lower_bound = lower_bound, candidate_vars = [i for i in range(model.NumVars)])
candidate_node = [root_node]
current_optimum = None

while candidate_node:
    node, candidate_node = choice_node(candidate_node)
    if node.upper_bound <= lower_bound:
        print("prune by bound")
        continue
    model_status = node.optimize(heuristic_solve)
    if model_status == 'infeasible':
        print("prune by infeasiblity")
        continue
    node.update_upper_bound()
    if node.upper_bound <= lower_bound:
        print("prune by bound")
        continue
    if node.is_integer():
        node.update_lower_bound()
        if node.lower_bound > lower_bound:
            lower_bound = node.lower_bound
            current_optimum = node.solution
        continue
    if node.is_child_problem():
        child_node1, child_node2 = node.get_child_problem()
        candidate_node.append(child_node1)
        candidate_node.append(child_node2)

print("lower_bound: ", lower_bound)
print("optimum:", current_optimum)

    


    

