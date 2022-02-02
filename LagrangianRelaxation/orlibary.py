
import numpy as np
class ORLibary:
    def __init__(self):
        self.route = './data\\'

    def gen_assign_prob(self, file_name):
        with open(self.route + file_name, 'r') as f:
            gen_assign_prob = f.read()

        gen_assign_prob = gen_assign_prob.strip().split('\n')
        size = gen_assign_prob[0].strip().split(' ')
        n_agent, n_job = int(size[0]), int(size[1])
        capacity = gen_assign_prob[-1].strip().split(' ')[0]
        b, A = [int(capacity) for i in range(n_agent)], []
        for i in range(1, len(gen_assign_prob )):
            temp = gen_assign_prob [i].strip().split(' ')
            temp = [int(x) for x in temp]
            for x in temp:
                A.append(x)
            if len(A) == 2 * n_agent * n_job:
                break
            if len(A) > 2 * n_agent * n_job:
                print("constraints data error")

        A = np.array(A).reshape((2*n_agent, n_job))
        cost_coeff = A[0:n_agent,:]
        A = A[n_agent:,:]
        return n_agent, n_job, A, b, cost_coeff