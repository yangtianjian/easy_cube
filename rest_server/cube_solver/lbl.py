import numpy as np
from abc import ABCMeta, abstractmethod
from collections import deque


def get_face_map():
    return {'D': 0, 'R': 1, 'B': 2, 'U': 3, 'L': 4, 'F': 5}

def get_face_rel():
    '''
    Get the relation of faces. The orientation of the cube should be 123456789 from top to down, left to right
    :return: Two dicts. The face letter and its adjacent faces plus the index of adjacent faces
    The neighbors are stated clockwise, respect to the face in front of you
    '''
    return {
        'F': {'L': 'L', 'R': 'R', 'U': 'U', 'D': 'D'},
        'L': {'L': 'B', 'R': 'F', 'U': 'U', 'D': 'D'},
        'R': {'L': 'F', 'R': 'B', 'U': 'U', 'D': 'D'},
        'U': {'L': 'L', 'R': 'R', 'U': 'B', 'D': 'F'},
        'D': {'L': 'L', 'R': 'R', 'U': 'F', 'D': 'B'},
        'B': {'L': 'R', 'R': 'L', 'U': 'U', 'D': 'D'}
    }, {
        'F': {'L': [8, 5, 2], 'U': [6, 7, 8], 'R': [0, 3, 6], 'D': [2, 1, 0]},
        'L': {'L': [8, 5, 2], 'U': [0, 3, 6], 'R': [0, 3, 6], 'D': [0, 3, 6]},
        'R': {'L': [8, 5, 2], 'U': [8, 5, 2], 'R': [0, 3, 6], 'D': [8, 5, 2]},
        'U': {'L': [2, 1, 0], 'U': [2, 1, 0], 'R': [2, 1, 0], 'D': [2, 1, 0]},
        'D': {'L': [6, 7, 8], 'U': [6, 7, 8], 'R': [6, 7, 8], 'D': [6, 7, 8]},
        'B': {'L': [8, 5, 2], 'U': [2, 1, 0], 'R': [0, 3, 6], 'D': [6, 7, 8]}
    }

def print_(x):
    print(x, end='')


class CubeAnswer(object):

    def __init__(self):
        '''
        The answer format of the rubik's cube
        '''
        self.answer_str = ""
        self.info = {}

class Cube(object):
    '''
    The faces should be unfolded like this:
     U
    LFRB
     D
        012
        345
        678
     012012012012
     345345345345
     678678678678
        012
        345
        678
    '''

    def __init__(self, initial_color=None):
        '''
        Initialize a cube with ndarray or cube string.
        :param initial_color: str describe the cube from DRBULF, row-wise
        '''
        self._face_map = get_face_map()
        self._face_rel, self._nb = get_face_rel()  # The relation of face and the index of affected blocks when turning

        if initial_color is None:
            self._c = np.stack([np.ones(shape=(3, 3), dtype=np.int) * i for i in range(6)], axis=0)
        elif isinstance(initial_color, str):
            if len(initial_color) != 54:
                raise ValueError("The cube string must have shape of 54")
            self._c = np.array([self._face_map[x] for x in initial_color]).reshape((6, 3, 3))
        elif isinstance(initial_color, np.ndarray):
            self._c = initial_color.reshape((6, 3, 3))
        else:
            raise ValueError("The cube string must be either string or ndarray")

    def __copy__(self):
        c = Cube()
        c._face_map = self._face_map
        c._face_rel = self._face_rel
        c._c = self._c.copy()   # Only copy the state of the cube because others are read-only
        return c

    def copy(self):
        return self.__copy__()


    def rotate(self, face, clockwise=None):

        if clockwise is None and len(face) == 1:
            clockwise = True

        if len(face) == 2:
            if face[1] == "'":
                if clockwise is None or not clockwise:
                    clockwise = False
                else:
                    raise ValueError("The clockwise is True, but the opeation has ' inside")
            face = face[0]

        v = self._c.reshape(6, 9)   # This view shares the memory with the original cube, making it easy to operate
        f_idx = self._face_map[face] if isinstance(face, str) else face
        l_idx, r_idx, u_idx, d_idx = [self._face_map[self._face_rel[face][f]] for f in ['L', 'R', 'U', 'D']]
        l_nbr, r_nbr, u_nbr, d_nbr = [self._nb[face][f] for f in ['L', 'R', 'U', 'D']]

        temp = v[l_idx, l_nbr].copy()
        if clockwise:
            self._c[f_idx] = (self._c[f_idx].T)[:, ::-1]   # rotate the whole face as 2D-array 90 degree clockwise
            v[l_idx, l_nbr] = v[d_idx, d_nbr]
            v[d_idx, d_nbr] = v[r_idx, r_nbr]
            v[r_idx, r_nbr] = v[u_idx, u_nbr]
            v[u_idx, u_nbr] = temp
        else:
            self._c[f_idx] = (self._c[f_idx].T)[::-1, :]
            v[l_idx, l_nbr] = v[u_idx, u_nbr]
            v[u_idx, u_nbr] = v[r_idx, r_nbr]
            v[r_idx, r_nbr] = v[d_idx, d_nbr]
            v[d_idx, d_nbr] = temp
        return self

    def rotate_sequence(self, seq):
        i = 0
        s = []
        while i < len(seq):
            t = seq[i]
            if i < len(seq) - 1 and seq[i + 1] == "'":
                t += "'"
                i += 1
            s.append(t)
            i += 1
        for r in s:
            self.rotate(r)
        return self

    def __getitem__(self, item):
        if item in ['L', 'R', 'U', 'D', 'F', 'B']:
            return self._c[self._face_map[item]]

    def __str__(self):
        s = ""

        fu = self['U']
        for i in range(3):
            s += '   '
            for j in range(3):
                s += str(fu[i][j])
            s += "\n"

        f = ['L', 'F', 'R', 'B']
        for i in range(3):
            for j in f:
                for k in range(3):
                    s += str(self[j][i, k])
            s += "\n"

        fd = self['D']
        for i in range(3):
            s += '   '
            for j in range(3):
                s += str(fd[i][j])
            s += "\n"
        return s

    def __eq__(self, other):
        return np.all(self._c == other._c)

    def __hash__(self):
        sum = 0
        d = 0
        im = 2 ** 60
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    sum += ((6 ** d) % (im)) * int(self._c[i, j, k])
                    d += 1
        return sum % im


class CubeSolver(metaclass=ABCMeta):

    def __init__(self):
        super(CubeSolver, self).__init__()

    @abstractmethod
    def solve(self, cube: Cube):
        pass

class BruteSolver(CubeSolver):

    def __init__(self):
        super().__init__()
        self.terminal = Cube()

    def solve(self, cube: Cube):
        q = deque()
        h = set()
        q.appendleft({"c": cube, "method": ""})
        h.add(cube)

        if cube == self.terminal:
            return ""

        methods = ["L", "R", "U", "D", "F", "B", "L'", "R'", "U'", "D'", "F'", "B'"]
        while len(q) > 0:
            head = q.pop()
            for method in methods:
                cube0 = head['c'].__copy__().rotate(method)
                if cube0 == self.terminal:
                    return head['method'] + method
                if cube0 not in h:
                    h.add(cube0)
                    q.appendleft({"c": cube0, "method": head['method'] + method})
        return "No answer"

class LBLSolver(CubeSolver):

    def __init__(self):
        super(LBLSolver, self).__init__()
        self.terminal = Cube()
        self._face_map = get_face_map()
        self._face_rel, self._nb = get_face_rel()

    def solve_up_cross(self, cube):
        pass

    def solve_up_corner(self, cube):
        pass

    def solve_middle_layer(self, cube):
        pass

    def solve_yellow_cross(self, cube):
        pass

    def solve:


    def solve(self, cube: Cube):




if __name__ == '__main__':
    cube = Cube()
    cube.rotate_sequence("DDRUL'")
    ans = BruteSolver().solve(cube)
    print(ans)