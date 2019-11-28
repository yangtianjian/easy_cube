import numpy as np
from abc import ABCMeta, abstractmethod
from collections import deque
import kociemba
import pycuber as pc
from deprecated import deprecated
from tqdm import tqdm, trange


def get_face_map():
    return {'D': 0, 'R': 1, 'B': 2, 'U': 3, 'L': 4, 'F': 5}

def get_inv_face_map():
    fm = get_face_map()
    return dict([(v, k) for k, v in fm.items()])

def get_kociemba_face_map():
    return {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}

def get_inv_kociemba_face_map():
    fm = get_kociemba_face_map()
    return dict([(v, k) for k, v in fm.items()])

def get_lbl_wrapper_face_map():
    return {'U': 0, 'F': 1, 'B': 2, 'R': 3, 'L': 4, 'D': 5}

def get_lbl_wrapper_color_map():
    return {'U': 'w', 'F': 'r', 'L': 'g', 'R': 'b', 'B': 'o', 'D': 'y'}

def get_inv_lbl_wrapper_color_map():
    cm = get_lbl_wrapper_color_map()
    return dict([(v, k) for k, v in cm.items()])

def get_inv_lbl_wrapper_face_map():
    fm = get_lbl_wrapper_face_map()
    return dict([(v, k) for k, v in fm.items()])

def shortcut(op, times):
    if times <= 2:
        return [op] * times
    if times == 3:
        if op[-1] == "'":
            return [op[0]] * (4 - times)
        else:
            return [op + "'"]
    return []

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
    _face_map = get_face_map()
    _inv_face_map = get_inv_face_map()
    _face_rel, _nb = get_face_rel()
    _kociemba_face_map = get_kociemba_face_map()
    _inv_kociemba_face_map = get_inv_kociemba_face_map()
    _lbl_face_map = get_lbl_wrapper_face_map()
    _inv_lbl_face_map = get_inv_lbl_wrapper_face_map()
    _lbl_color_map = get_lbl_wrapper_color_map()
    _inv_lbl_color_map = get_inv_lbl_wrapper_color_map()


    def __init__(self, initial_color=None):
        '''
        Initialize a cube with ndarray or cube string.
        :param initial_color: str describe the cube from DRBULF, row-wise
        '''

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

        self._rec = []   # All operations done to the cube, including turning the whole cube and for one step.
        self._op_label = {}  # The operation span. [string => tuple]

    def __copy__(self):
        '''
        A deep copy of the cube object
        :return: A copied object
        '''
        c = Cube()
        c._c = self._c.copy()   # Only copy the state of the cube because others are read-only
        c._rec = self._rec.copy()
        c._op_label = self._op_label.copy()
        return c

    def clear_record(self):
        self._rec.clear()
        self._op_label.clear()

    def get_record(self):
        return self._rec

    def get_array(self):
        return self._c

    def push_back_record(self, op):
        if len(op) == 0:  # empty string or empty list
            return
        if isinstance(op, str):
            self._rec.append(op)
        elif isinstance(op, list):
            self._rec.extend(op)
        else:
            raise ValueError("The record should either be string or list")

    def to_kociemba_compatible_string(self):
        '''
        The kociemba library has different face map. We have to do rearrangement and flatten the array to string.
        :return: Kociemba representation of the cube.
        '''
        c2 = self._c.reshape(6, 9)
        s = ''
        for i in range(6):
            colors = c2[self._face_map[self._inv_kociemba_face_map[i]], :]
            s += ''.join([self._inv_face_map[c] for c in colors])
        return s

    @deprecated
    def to_lbl_wrapper_compatible_string(self):
        '''
        It is for lbl solver. But this wrapper does not seem to work even in some simple cases. Therefore, I'll write the
        rules for myself. In case of further reuse, it is kept.
        :return: LBL solver-compatible string input.
        '''
        ans_str = ""
        cr = self._c.reshape(6, 9)
        for i in range(6):
            face_name_in_cpp = self._inv_lbl_face_map[i]
            face_idx_in_py = self._face_map[face_name_in_cpp]
            if face_name_in_cpp == 'U':
                perm = [5, 8, 7, 6, 3, 0, 1, 2, 4]
            elif face_name_in_cpp == 'D':
                perm = [5, 2, 1, 0, 3, 6, 7, 8, 4]
            else:
                perm = [2, 5, 8, 7, 6, 3, 0, 1, 4]
            seq = "".join([self._lbl_color_map[self._inv_face_map[x]] for x in cr[face_idx_in_py][perm]])
            ans_str += seq
        return ans_str

    def copy(self):
        return self.__copy__()

    def _rotate_xy_middle(self):
        '''
        Rotate the 8 cubes of the center which is on xy-plane. This is only used for rotate the whole cube conveniently.
        :return: None
        '''
        f_idx, r_idx, b_idx, l_idx = [self._face_map[f] for f in ['F', 'R', 'B', 'L']]
        v = self._c.reshape(6, 9)
        tmp = v[f_idx, [3, 4, 5]].copy()
        v[f_idx, [3, 4, 5]] = v[r_idx, [3, 4, 5]]
        v[r_idx, [3, 4, 5]] = v[b_idx, [3, 4, 5]]
        v[b_idx, [3, 4, 5]] = v[l_idx, [3, 4, 5]]
        v[l_idx, [3, 4, 5]] = tmp

    def _rotate_yz_middle(self):
        f_idx, d_idx, b_idx, u_idx = [self._face_map[f] for f in ['F', 'D', 'B', 'U']]
        v = self._c.reshape(6, 9)
        tmp = v[f_idx, [1, 4, 7]].copy()
        v[f_idx, [1, 4, 7]] = v[u_idx, [1, 4, 7]]
        v[u_idx, [1, 4, 7]] = v[b_idx, [7, 4, 1]]
        v[b_idx, [1, 4, 7]] = v[d_idx, [7, 4, 1]]  # this one is special!
        v[d_idx, [1, 4, 7]] = tmp

    def _rotate_xz_middle(self):
        u_idx, r_idx, d_idx, l_idx = [self._face_map[f] for f in ['U', 'R', 'D', 'L']]
        v = self._c.reshape(6, 9)
        tmp = v[u_idx, [3, 4, 5]].copy()
        v[u_idx, [3, 4, 5]] = v[l_idx, [7, 4, 1]]
        v[l_idx, [7, 4, 1]] = v[d_idx, [5, 4, 3]]
        v[d_idx, [5, 4, 3]] = v[r_idx, [1, 4, 7]]  # this one is special!
        v[r_idx, [1, 4, 7]] = tmp

    def rotate(self, face, clockwise=None, record=True):
        '''
        Operate the cube
        :param face: Which face has to be turned (clockwise or counterclockwise)
        :param clockwise: Whether it is clockwise
        :param record: Whether the rotation should be recorded into the _rec. The whole cube rotation should not be recorded.
        :return: None
        '''

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
            if record:
                self._rec.append(face)
        else:
            self._c[f_idx] = (self._c[f_idx].T)[::-1, :]
            v[l_idx, l_nbr] = v[u_idx, u_nbr]
            v[u_idx, u_nbr] = v[r_idx, r_nbr]
            v[r_idx, r_nbr] = v[d_idx, d_nbr]
            v[d_idx, d_nbr] = temp
            if record:
                self._rec.append(face[0] + "'")

        return self

    def _parse_str_seq_to_list(self, seq):
        i = 0
        s = []
        while i < len(seq):
            t = seq[i]
            if i < len(seq) - 1:
                if seq[i + 1] == "'":
                    t += "'"
                    i += 1
                elif seq[i + 1] == "2":
                    t += '2'
                    i += 1
            s.append(t)
            i += 1
        return s

    def rotate_sequence(self, seq, record=True):
        '''
        Perform a set of operations represented by string. Support ' or 2, but do not support some like "L'2", "L2'"
        :param seq: The operation sequence.
        :return: None. But the cube array has been modified.
        '''
        if len(seq) == 0:
            return
        if isinstance(seq, str):
            seq = self._parse_str_seq_to_list(seq)
        for r in seq:
            if r[-1] == '2':
                self.rotate(r[0], record=record)
                self.rotate(r[0], record=record)
            else:
                self.rotate(r, record=record)
        return self

    def view(self, axis, record=True):
        '''
        We rotate the whole cube around an axis.
        X axis: A ray from left to right
        Y axis: A ray from front to back
        Z axis: A ray from down to up
        To determine clockwise or counterclockwise, we have to look at the direction of the ray.
        :param axis: The axis. The value can be x, y, z or x', y', z'. If it is 'o', the face direction will be recovered.
        :param record: Whether we rotate the "whole turning" operation.
        :return: None. But the rotation will be recorded, for further recovery.
        '''
        axis = axis.lower()
        clockwise = True
        if len(axis) > 1 and axis[1] == "'":
            clockwise = False
        if axis[0] == 'x':
            if clockwise:
                self.rotate_sequence("LR'", record=False)
                self._rotate_yz_middle()
                if record:
                    self._rec.append("x")
            else:
                self.rotate_sequence("L'R", record=False)
                self._rotate_yz_middle()
                self._rotate_yz_middle()
                self._rotate_yz_middle()
                if record:
                    self._rec.append("x'")
        elif axis[0] == 'y':
            if clockwise:
                self.rotate_sequence("FB'", record=False)
                self._rotate_xz_middle()
                if record:
                    self._rec.append("y")
            else:
                self.rotate_sequence("F'B", record=False)
                self._rotate_xz_middle()
                self._rotate_xz_middle()
                self._rotate_xz_middle()
                if record:
                    self._rec.append("y'")
        elif axis[0] == 'z':
            if clockwise:
                self.rotate_sequence("UD'", record=False)
                self._rotate_xy_middle()
                if record:
                    self._rec.append("z")
            else:
                self.rotate_sequence("U'D", record=False)
                self._rotate_xy_middle()
                self._rotate_xy_middle()
                self._rotate_xy_middle()
                if record:
                    self._rec.append("z'")
        elif axis[0] == 'o':
            inv_op = {
                "x": "x'",
                "y": "y'",
                "z": "z'",
                "x'": "x",
                "y'": "y",
                "z'": "z"
            }
            for r in reversed(self._rec):
                if r in inv_op.keys():
                    self.view(inv_op[r], record) # The inverse operation may also be recorded.

        else:
            return ValueError("axis = {}, which has illegal letter".format(axis))

    def is_up_cross_finished(self):
        flag = True
        cp = self.copy()
        cp.view("o", record=False)
        for j in range(4):
            if not (cp['U'][2][1] == cp['U'][1][1] and cp['F'][0][1] == cp['F'][1][1]):
                flag = False
                break
            cp.view('z', record=False)
        return flag

    def is_up_corner_finished(self):
        flag = True
        cp = self.copy()
        cp.view("o", record=False)
        for j in range(4):
            if not (cp['U'][2][0] == cp['U'][1][1] and cp['L'][0][2] == cp['L'][1][1] and
                    cp['F'][0][0] == cp['F'][1][1]):
                flag = False
                break
            cp.view('z', record=False)
        return flag

    def is_middle_layer_finished(self):
        flag = True
        cp = self.copy()
        cp.view("o", record=False)
        for j in range(4):
            if not (cp['F'][1][0] == cp['F'][1][1] and cp['L'][1][2] == cp['L'][1][1]):
                flag = False
                break
            cp.view('z', record=False)
        return flag

    def is_down_cross_finished(self):
        cp = self.copy()
        cp.view("o", record=False)
        return cp['D'][0][1] == cp['D'][1][0] == cp['D'][1][1] == cp['D'][1][2] == cp['D'][2][1]

    def is_down_corner_finished(self):
        cp = self.copy()
        cp.view("o", record=False)
        return cp['D'][0][0] == cp['D'][0][2] == cp['D'][2][0] == cp['D'][2][2]


    def __getitem__(self, item):
        if item in ['L', 'R', 'U', 'D', 'F', 'B']:
            return self._c[self._face_map[item]]

    def __str__(self):
        s = ""

        fu = self['U']
        for i in range(3):
            s += '         '
            for j in range(3):
                s += str(fu[i][j]) + "  "
            s += "\n"

        f = ['L', 'F', 'R', 'B']
        for i in range(3):
            for j in f:
                for k in range(3):
                    s += str(self[j][i, k]) + "  "
            s += "\n"

        fd = self['D']
        for i in range(3):
            s += '         '
            for j in range(3):
                s += str(fd[i][j]) + "  "
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
    _face_map = get_face_map()
    _inv_face_map = get_inv_face_map()
    _face_rel, _nb = get_face_rel()

    def __init__(self):
        super(CubeSolver, self).__init__()

    @abstractmethod
    def solve(self, cube: Cube):
        pass


class BruteSolver(CubeSolver):

    def __init__(self, terminal=Cube()):
        super().__init__()
        self.terminal = terminal

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

    # For up cross
    def _case1(self, cube, desire):
        for i in range(4):  # U on the left
            if cube['L'][2][1] == self._face_map['U'] and cube['D'][1][0] == desire:
                cube.push_back_record(shortcut('L', i))
                cube.rotate_sequence("L'FL")
                cube.rotate_sequence(shortcut("L'", i))  # avoid messing up
                return True
            cube.rotate('L', record=False)   # For covering other cases
        return False

    def _case2(self, cube, desire):
        for i in range(4): # U on the right
            if cube['R'][2][1] == self._face_map['U'] and cube['D'][1][2] == desire:
                cube.push_back_record(shortcut('R', i))
                cube.rotate_sequence("RF'R'")
                cube.rotate_sequence(shortcut("R'", i))  # avoid messing up
                return True
            cube.rotate('R', record=False)   # For covering other cases
        return False

    def _case3(self, cube, desire):
        for i in range(4): # U on the front
            if cube['F'][2][1] == self._face_map['U'] and cube['D'][0][1] == desire:
                cube.push_back_record(shortcut('F', i))
                cube.rotate_sequence("DRF'R'")
                return True
            cube.rotate('F', record=False)   # For covering other cases
        return False

    def _case4(self, cube, desire):
        for i in range(4): # U on the back
            if cube['B'][2][1] == self._face_map['U'] and cube['D'][2][1] == desire:
                # We switch the camera to make it more clear
                cube.push_back_record(shortcut('B', i))
                cube.rotate_sequence("D")
                cube.rotate_sequence(shortcut("B'", i))  # prevent from messing the others
                cube.rotate_sequence("L'FL")
                return True
            cube.rotate('B', record=False)   # For covering other cases
        return False

    def _case5(self, cube, desire):
        for i in range(4): # U on the down side
            if cube['D'][0][1] == self._face_map['U'] and cube['F'][2][1] == desire:
                cube.push_back_record(shortcut('D', i))
                cube.rotate_sequence("FF")
                return True
            cube.rotate('D', record=False)
        return False

    # For up corners
    def _case6(self, cube, desire_f, desire_l):
        '''
        可以通过D操作达到c['L'][2][2] = 'U' 且的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        for i in range(4):
            if cube['L'][2][2] == self._face_map['U'] and cube['F'][2][0] == desire_f and \
                    cube['D'][0][0] == desire_l:
                cube.push_back_record(shortcut('D', i))
                cube.rotate_sequence("D'F'DF")
                return True
            cube.rotate('D', record=False)
        return False

    def _case7(self, cube, desire_f, desire_l):
        '''
        可以通过D操作达到c['F'][2][0] = 'U' 且的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        for i in range(4):
            if cube['F'][2][0] == self._face_map['U'] and cube['L'][2][2] == desire_l and \
                    cube['D'][0][0] == desire_f:
                cube.push_back_record(shortcut('D', i))
                cube.rotate_sequence("DLD'L'")
                return True
            cube.rotate('D', record=False)
        return False

    def _case8(self, cube, desire_f, desire_l):
        '''
        经过D操作可以将U带到c['D'][0][0]
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        for i in range(4):
            if cube['D'][0][0] == self._face_map['U'] and cube['L'][2][2] == desire_f and \
                    cube['F'][2][0] == desire_l:
                cube.push_back_record(shortcut('D', i))
                cube.rotate_sequence("F'D'D'FD")
                self._case7(cube, desire_f, desire_l)
                return True
            cube.rotate('D', record=False)
        return False

    def _case9(self, cube, desire_f, desire_l):
        '''
        c['L'][0][2] = 'U' and c['F'][0][0] = desire_l and c['U'][2][0] = desire_f
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['L'][0][2] == self._face_map['U'] and cube['F'][0][0] == desire_l and cube['U'][2][0] == desire_f:
            cube.rotate_sequence("LDL'D'")
            return self._case6(cube, desire_f, desire_l)


    def _case10(self, cube, desire_f, desire_l):
        '''
        对角调换的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['U'][0][2] == self._face_map['U'] and cube['R'][0][2] == desire_l and cube['B'][0][0] == desire_f:
            cube.rotate_sequence("B'D'D'B")
            return self._case6(cube, desire_f, desire_l)

    def _case11(self, cube, desire_f, desire_l):
        '''
        与后面的邻角调换的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['U'][0][0] == self._face_map['U'] and cube['L'][0][0] == desire_f and cube['B'][0][2] == desire_l:
            cube.rotate_sequence("L'DLD")
            return self._case7(cube, desire_f, desire_l)

    def _case12(self, cube, desire_f, desire_l):
        '''
        与右面的邻角调换的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['U'][2][2] == self._face_map['U'] and cube['F'][0][2] == desire_l and cube['R'][0][0] == desire_f:
            cube.rotate_sequence("R'D'R")
            return self._case6(cube, desire_f, desire_l)

    def _case13(self, cube, desire_f, desire_l):
        '''
        U在FR上排的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['F'][0][2] == self._face_map['U'] and cube['U'][2][2] == desire_f and cube['R'][0][0] == desire_l: # 右面邻角的前面
            cube.rotate_sequence("R'D'R")
            return self._case8(cube, desire_f, desire_l)
        if cube['R'][0][0] == self._face_map['U'] and cube['F'][0][2] == desire_f and cube['U'][2][2] == desire_l: # 右面邻角的右面
            cube.rotate_sequence("R'D'R")
            return self._case7(cube, desire_f, desire_l)

    def _case14(self, cube, desire_f, desire_l):
        '''
        U在FL上排的情况
        :param cube:
        :param desire_f:
        :param desire_l:
        :return:
        '''
        if cube['F'][0][0] == self._face_map['U'] and cube['U'][2][0] == desire_l and cube['L'][0][2] == desire_f:
            cube.rotate_sequence("LDL'D'")
            return self._case8(cube, desire_f, desire_l)

    def _case15(self, cube, desire_f, desire_l):  # U在BR上排的情况
        if cube['B'][0][0] == self._face_map['U'] and cube['U'][0][2] == desire_l and cube['R'][0][2] == desire_f:
            cube.rotate_sequence("B'D'D'B")
            return self._case7(cube, desire_f, desire_l)
        if cube['R'][0][2] == self._face_map['U'] and cube['B'][0][0] == desire_l and cube['U'][0][2] == desire_f:
            cube.rotate_sequence("B'D'D'B")
            return self._case8(cube, desire_f, desire_l)

    def _case16(self, cube, desire_f, desire_l):  # U在BL上排的情况
        if cube['B'][0][2] == self._face_map['U'] and cube['L'][0][0] == desire_l and cube['U'][0][0] == desire_f:
            cube.rotate_sequence("BDB'")
            return self._case6(cube, desire_f, desire_l)
        if cube['L'][0][0] == self._face_map['U'] and cube['U'][0][0] == desire_l and cube['B'][0][2] == desire_f:
            cube.rotate_sequence("L'DDLD'")
            return self._case7(cube, desire_f, desire_l)

    # For the middle layer
    def _case17(self, cube, desire_f, desire_l):
        # 上换左
        for i in range(4):
            if cube['F'][0][1] == desire_f and cube['U'][2][1] == desire_l:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("U'L'ULUFU'F'")  # The order of this formula is 15
                return True
            cube.rotate("U", record=False)
        return False

    def _case18(self, cube, desire_f, desire_l):
        # 上换右
        for i in range(4):
            if cube['U'][1][0] == desire_f and cube['L'][0][1] == desire_l:
                cube.push_back_record(shortcut('U', i))
                cube.view("z'")
                cube.rotate_sequence("URU'R'U'F'UF")  # The order of this formula is 15
                cube.view("z")
                return True
            cube.rotate("U", record=False)
        return False

    def _case19(self, cube, desire_f, desire_l):
        # 中层反位的情况
        if cube['L'][1][2] == desire_f and cube['F'][1][0] == desire_l:
            cube.view("z'")
            cube.rotate_sequence("URU'R'U'F'UF")  # We use formula in case 17 or 18
            cube.view("z")
            if self._case17(cube, desire_f, desire_l):
                return True
            if self._case18(cube, desire_f, desire_l):
                return True
            raise NotImplementedError("Please check implementation")
        return False

    # For the down cross
    def _case20(self, cube):
        # 小拐角的情况
        for i in range(4):
            if cube['U'][1][0] == cube['U'][1][1] and cube['U'][0][1] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("FURU'R'F'")
                return True
            cube.rotate("U", record=False)
        return False

    def _case21(self, cube):
        # 一字马的情况
        for i in range(4):
            if cube['U'][1][0] == cube['U'][1][1] and cube['U'][1][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("FURU'R'F'")  # 规约为拐角情况
                return self._case20(cube)
            cube.rotate("U", record=False)
        return False

    def _case22(self, cube):
        # 一个黄点的情况，上面一定是黄色的，不用匹配
        cube.rotate_sequence("FURU'R'F'")
        return self._case21(cube)  # 规约为一字马的情况

    # For the down corner, see https://zhuanlan.zhihu.com/p/42299115
    def _case23(self, cube):  # 右手小鱼
        for i in range(4):
            if cube['U'][2][0] == cube['F'][0][2] == cube['B'][0][2] == cube['R'][0][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return True
            cube.rotate("U", record=False)
        return False

    def _case24(self, cube):  # 左手小鱼
        for i in range(4):
            if cube['U'][2][2] == cube['F'][0][0] == cube['B'][0][0] == cube['L'][0][0] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("L'U'LU'L'U'U'L")
                return True
            cube.rotate("U", record=False)
        return False

    def _case25(self, cube):  # 左上右下都有
        for i in range(4):
            if cube['U'][0][0] == cube['U'][2][2] == cube['F'][0][0] == cube['R'][2][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return self._case24(cube)
            cube.rotate("U", record=False)
        return False

    def _case26(self, cube):  # 左上右上
        for i in range(4):
            if cube['U'][0][0] == cube['F'][0][0] == cube['F'][0][2] == cube['U'][0][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return self._case24(cube)
            cube.rotate("U", record=False)
        return False

    def _case27(self, cube):  # 右上右下
        for i in range(4):
            if cube['F'][0][0] == cube['U'][0][2] == cube['U'][2][2] == cube['B'][0][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return self._case24(cube)
            cube.rotate("U", record=False)
        return False

    def _case28(self, cube):  # 只有十字1
        for i in range(4):
            if cube['U'][2][2] == cube['F'][0][0] == cube['B'][0][0] == cube['L'][0][0] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return self._case23(cube)
            cube.rotate("U", record=False)
        return False

    def _case29(self, cube):  # 只有十字2
        for i in range(4):
            if cube['U'][2][0] == cube['F'][0][2] == cube['B'][0][2] == cube['R'][0][2] == cube['U'][1][1]:
                cube.push_back_record(shortcut('U', i))
                cube.rotate_sequence("RUR'URUUR'")
                return self._case23(cube)
            cube.rotate("U", record=False)
        return False


    def up_cross_one(self, cube, desire, stage):
        '''
        将一个UF edge 进行还原。
        :param cube:
        :return:
        '''
        if cube['U'][2][1] == self._face_map['U'] and cube['F'][0][1] == desire:
            return    # No operation needed.

        if self._case1(cube, desire):
            return

        if self._case2(cube, desire):
            return

        if self._case3(cube, desire):
            return

        if self._case4(cube, desire):
            return

        if self._case5(cube, desire):
            return

        # 特判U在上面
        if cube['U'][1][0] == self._face_map['U'] and cube['L'][0][1] == desire:  # 一定在cross的第一次发生
            assert(stage == 0)
            cube.rotate_sequence("U'")
            return
        if cube['U'][1][2] == self._face_map['U'] and cube['R'][0][1] == desire:
            cube.rotate_sequence("R'")
            if self._case3(cube, desire):
                return
        if cube['U'][0][1] == self._face_map['U'] and cube['B'][0][1] == desire:
            cube.rotate_sequence("B'")
            if self._case2(cube, desire):
                return

        raise NotImplementedError("There are other cases not implemented")

    def up_corner_one(self, cube, desire_f, desire_l):
        if cube['U'][2][0] == self._face_map['U'] and cube['L'][0][2] == desire_l and cube['F'][0][0] == desire_f:   # No operation needed
            return

        if self._case6(cube, desire_f, desire_l):
            return

        if self._case7(cube, desire_f, desire_l):
            return

        if self._case8(cube, desire_f, desire_l):
            return

        if self._case9(cube, desire_f, desire_l):
            return

        if self._case10(cube, desire_f, desire_l):
            return

        if self._case11(cube, desire_f, desire_l):
            return

        if self._case12(cube, desire_f, desire_l):
            return

        if self._case13(cube, desire_f, desire_l):
            return

        if self._case14(cube, desire_f, desire_l):
            return

        if self._case15(cube, desire_f, desire_l):
            return

        if self._case16(cube, desire_f, desire_l):
            return

        print(cube)
        raise NotImplementedError("Some top corner cases are not implemented")

    def solve_up_cross(self, cube):
        '''
        这里的思路是先还原一个UF，然后转魔方，以此类推，还原出四个UF
        :param cube:
        :return:
        '''
        for i in range(4):
            self.up_cross_one(cube, cube['F'][1][1], i)
            cube.view('z')

    def solve_up_corner(self, cube):
        '''
        这里的思路是还原ULF，然后转魔方
        :param cube:
        :return:
        '''
        for i in range(4):
            desire_l = cube['L'][1][1]
            desire_f = cube['F'][1][1]
            self.up_corner_one(cube, desire_f, desire_l)
            cube.view('z')

    def solve_middle_layer(self, cube):
        fail_cnt = 0    # If no operation happens after 4 rotations, we will know that some conditions are not covered
        while not cube.is_middle_layer_finished():
            if not (cube['L'][1][2] == cube['L'][1][1] and cube['F'][1][0] == cube['F'][1][1]):
                b1 = self._case17(cube, cube['F'][1][1], cube['L'][1][1])
                b2 = False
                b3 = False
                if not b1:
                    b2 = self._case18(cube, cube['F'][1][1], cube['L'][1][1])
                if not b1 and not b2:
                    b3 = self._case19(cube, cube['F'][1][1], cube['L'][1][1])
                if not b1 and not b2 and not b3:
                    fail_cnt += 1
                else:
                    fail_cnt = 0
            cube.view("z")
            if fail_cnt == 4:
                print(cube)
                raise NotImplementedError("Some condition may not be implemented in solving middle layer")

    def solve_down_cross(self, cube):
        if self._case20(cube):
            return True
        if self._case21(cube):
            return True
        if self._case22(cube):
            return True
        return False

    def solve_down_corner(self, cube):
        if self._case23(cube):
            return True
        if self._case24(cube):
            return True
        if self._case25(cube):
            return True
        if self._case26(cube):
            return True
        if self._case27(cube):
            return True
        if self._case28(cube):
            return True
        if self._case29(cube):
            return True
        return False

    def solve_down_corner_2(self, cube):
        pass

    def solve_down_edge(self, cube):
        pass

    def solve(self, cube: Cube, inplace=False, steps=99):
        if not inplace:
            cube = cube.copy()
        cube.clear_record()

        if steps >= 1:
            self.solve_up_cross(cube)
        if steps >= 2:
            self.solve_up_corner(cube)
        if steps >= 3:
            cube.view("x")
            cube.view("x")
            self.solve_middle_layer(cube)
        if steps >= 4:
            self.solve_down_cross(cube)
        if steps >= 5:
            self.solve_down_corner(cube)
        if steps >= 6:
            self.solve_down_corner_2(cube)
        if steps >= 7:
            self.solve_down_edge(cube)


class KociembaSolver(CubeSolver):
    def __init__(self, terminal=Cube(), single_step_format=False):
        '''
        Solve the cube to the terminal state
        :param terminal: The terminal cube state
        :param single_step_format: Whether to allow 180 degree turns in a single step. default False
        '''
        super(KociembaSolver, self).__init__()
        self.terminal = terminal
        self._single_step_format = single_step_format

    def solve(self, cube: Cube):
        if cube == self.terminal:
            return ""
        ans = kociemba.solve(cube.to_kociemba_compatible_string(), self.terminal.to_kociemba_compatible_string())
        if not self._single_step_format:
            tmp = list(ans)
            for i in range(1, len(tmp)):
                if tmp[i] == '2':
                    tmp[i] = tmp[i - 1]
            return "".join(tmp)
        return ans

# These are functions for unit test
def create_random_cube(seed, return_op=False):
    np.random.seed(seed)
    cube = Cube()
    q = np.random.randint(12, 20)
    seq = []
    for i in range(q):
        step = ['U', 'D', 'L', 'R', 'F', 'B'][np.random.randint(0, 6)]
        b = np.random.randint(0, 2)
        if b == 1:
            step += "'"
        seq.append(step)
    cube.rotate_sequence(seq, record=False)
    if not return_op:
        return cube
    else:
        return cube, seq

def is_valid_cube(cube):
    arr = cube.get_array()
    return np.all(np.bincount(arr.reshape((-1, ))) == 9)


def _ut_up_cross():
    T = 1000
    for i in trange(T):
        cube = create_random_cube(seed=41)
        cube_c = cube.copy()
        solver = LBLSolver()
        solver.solve(cube_c, inplace=True, steps=1)
        if not cube_c.is_up_cross_finished():
            print("=======Error Happens========")
            print(cube)
            print("=======See clearly==========")
            print(cube_c)
            break

def _ut_up_corner():
    T = 1000
    for i in trange(T):
        cube = create_random_cube(seed=43)
        cube_c = cube.copy()
        solver = LBLSolver()

        solver.solve(cube_c, inplace=True, steps=2)
        if not (cube_c.is_up_cross_finished() and cube_c.is_up_corner_finished() and is_valid_cube(cube_c)):
            print("=======Error Happens========")
            print(cube)
            print("=======See clearly==========")
            print(cube_c)
            break

def _ut_middle_layer():
    T = 1000
    for i in trange(T):
        cube, steps = create_random_cube(seed=43, return_op=True)
        cube_c = cube.copy()
        solver = LBLSolver()

        try:
            solver.solve(cube_c, inplace=True, steps=3)
            if not (cube_c.is_up_cross_finished() and cube_c.is_up_corner_finished() and cube_c.is_middle_layer_finished()):
                raise ValueError
        except ValueError as e:
            print("=======Error Happens========")
            print(str(e))
            print(cube)
            print("=======See clearly==========")
            for k in range(4):
                print(cube_c)
                cube_c.view("z", record=False)
                print("===========")
            print("=======Rotate_sequence======")
            print(" ".join(steps))
            print(" ".join(cube_c.get_record()))
            break


def _ut_bottom_cross():
    T = 1000
    for i in trange(T):
        cube, steps = create_random_cube(seed=43, return_op=True)
        cube_c = cube.copy()
        solver = LBLSolver()

        try:
            solver.solve(cube_c, inplace=True, steps=4)
            if not (cube_c.is_up_cross_finished() and cube_c.is_up_corner_finished() and cube_c.is_middle_layer_finished()
            and cube_c.is_down_cross_finished()):
                raise ValueError
        except ValueError as e:
            print("=======Error Happens========")
            print(str(e))
            print(cube)
            print("=======See clearly==========")
            for k in range(4):
                print(cube_c)
                cube_c.view("z", record=False)
                print("===========")
            print("=======Rotate_sequence======")
            print(" ".join(steps))
            print(" ".join(cube_c.get_record()))
            break

def _ut_bottom_corner():
    T = 1000
    for i in trange(T):
        cube, steps = create_random_cube(seed=43, return_op=True)
        cube_c = cube.copy()
        solver = LBLSolver()

        try:
            solver.solve(cube_c, inplace=True, steps=5)
            if not (cube_c.is_up_cross_finished() and cube_c.is_up_corner_finished() and cube_c.is_middle_layer_finished()
            and cube_c.is_down_cross_finished() and cube_c.is_down_corner_finished()):
                raise ValueError
        except ValueError as e:
            print("=======Error Happens========")
            print(str(e))
            print(cube)
            print("=======See clearly==========")
            for k in range(4):
                print(cube_c)
                cube_c.view("z", record=False)
                print("===========")
            print("=======Rotate_sequence======")
            print(" ".join(steps))
            print(" ".join(cube_c.get_record()))
            break

def _ut_basic_op():
    cube = create_random_cube(seed=43)
    print(cube)
    cube.view("x")
    print(cube)


def regression_test():
    # _ut_basic_op()
    _ut_up_cross()
    _ut_up_corner()
    _ut_middle_layer()
    _ut_bottom_cross()
    _ut_bottom_corner()

if __name__ == '__main__':
    regression_test()

# UFBRLD
# gwbgrgorw rybwoboyr yrybwrbgo ywgrwobyb wwywbbryg grooggooy
# gwbgrgorwrybwoboyryrybwrbgoywgrwobybwwywbbryggrooggooy