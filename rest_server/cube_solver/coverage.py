
from collections import defaultdict
from cube_solver.solver import create_random_cube, KociembaSolver, LBLSolver
from tqdm import tqdm, trange



if __name__ == '__main__':
    with open('./test_log.txt', 'r') as f:
        s = defaultdict(set)
        for line in f:
            line = line.strip().split("\t")
            c = line[3].split('/')
            prog, case = c[0], c[1]
            try:
                s[prog].add(int(case.split("_")[1]))
            except ValueError as e:
                continue
    print(s)