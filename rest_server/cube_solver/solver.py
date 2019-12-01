import numpy as np
from abc import ABCMeta, abstractmethod
from collections import deque
import kociemba
import pycuber as pc
from deprecated import deprecated
from tqdm import tqdm, trange
import json


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
        self._op_label = []  # The operation span. [string => tuple] e.g.  ("step", 0, 3)

        self._transaction_rec = []

    def _check_valid_with_kociemba(self):
        try:
            kociemba.solve(self.to_kociemba_compatible_string())
            return True
        except ValueError as e:
            return False

    @staticmethod
    def from_json(json_):
        if isinstance(json_, str):
            json_ = json.loads(json_)
        arr_ = np.stack([np.array(json_[x]).reshape(3, 3) for x in ['D', 'R', 'B', 'U', 'L', 'F']], axis=0)
        cube = Cube(initial_color=arr_)
        if not cube._check_valid_with_kociemba():
            raise ValueError("The cube is not valid")
        return cube

    def __copy__(self):
        '''
        A deep copy of the cube object
        :return: A copied object
        '''
        c = Cube()
        c._c = self._c.copy()   # Only copy the state of the cube because others are read-only
        c._rec = self._rec.copy()
        c._op_label = self._op_label.copy()
        c._transaction_rec = self._transaction_rec.copy()
        return c

    def new_transaction(self):
        self._transaction_rec = []

    def commit(self):
        self._rec.extend(self._transaction_rec)
        self._transaction_rec.clear()

    def get_transaction_cache(self):
        return self._transaction_rec.copy()

    def rollback(self):
        self._transaction_rec.clear()

    def clear_record(self):
        self._rec.clear()
        self._op_label.clear()
        self._transaction_rec.clear()

    def get_record(self):
        return self._rec.copy()

    def get_mirroring_record(self):

        '''
         U
        LFRB   => ULFRBD
         D
        :return:
        '''
        def mirror(arr, rot):
            mirror_map = {
                "x":  [4, 1, 0, 3, 5, 2],
                "x'": [2, 1, 5, 3, 0, 4],
                "y":  [1, 5, 2, 0, 4, 3],
                "y'": [3, 0, 2, 5, 4, 1],
                "z":  [0, 2, 3, 4, 1, 5],
                "z'": [0, 4, 1, 2, 3, 5]
            }
            return arr[mirror_map[rot]]

        ret = []
        mirror_map = ['U', 'L', 'F', 'R', 'B', 'D']
        inv_mirror_map = dict((k, i) for i, k in enumerate(mirror_map))
        cur_mirror = np.arange(6)
        for r in self._rec:
            if r[0] in ['x', 'y', 'z']:
                cur_mirror = mirror(cur_mirror, r)
                ret.append(r)
            else:
                trans_r = mirror_map[cur_mirror[inv_mirror_map[r[0]]]]
                if len(r) > 1 and r[1] == "'":
                    ret.append(trans_r + "'")
                else:
                    ret.append(trans_r)
        return ret

    def get_op_label(self):
        return self._op_label.copy()

    def get_array(self):
        return self._c.copy()

    def _push_back_record(self, op, label='Untitled'):
        if len(op) == 0:  # empty string or empty list
            return
        if not isinstance(op, list):
            if isinstance(op, str):
                op = self._parse_str_seq_to_list(op)
            else:
                raise ValueError("The record should either be string or list")
        self._rec.extend(op)
        if len(self._op_label) == 0:
            self._op_label.append((0, len(op) - 1, label))
        else:
            ll, lr, llb = self._op_label[-1]
            if label == llb:
                self._op_label[-1] = (ll, lr + len(op), label)
            else:
                self._op_label.append((lr + 1, lr + len(op), label))

    def _try_push_record(self, op, record=True):
        if record:
            if record is True:
                record = "Untitled"
            self._push_back_record(op, record)

    def pop_back_record(self):
        ll, lr, llb = self._op_label[-1]
        if ll == lr:
            self._op_label.pop()
        else:
            self._op_label[-1] = (ll, lr - 1, llb)
        return self._rec.pop()

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
        return self

    def _rotate_yz_middle(self):
        f_idx, d_idx, b_idx, u_idx = [self._face_map[f] for f in ['F', 'D', 'B', 'U']]
        v = self._c.reshape(6, 9)
        tmp = v[f_idx, [1, 4, 7]].copy()
        v[f_idx, [1, 4, 7]] = v[u_idx, [1, 4, 7]]
        v[u_idx, [1, 4, 7]] = v[b_idx, [7, 4, 1]]
        v[b_idx, [1, 4, 7]] = v[d_idx, [7, 4, 1]]  # this one is special!
        v[d_idx, [1, 4, 7]] = tmp
        return self

    def _rotate_xz_middle(self):
        u_idx, r_idx, d_idx, l_idx = [self._face_map[f] for f in ['U', 'R', 'D', 'L']]
        v = self._c.reshape(6, 9)
        tmp = v[u_idx, [3, 4, 5]].copy()
        v[u_idx, [3, 4, 5]] = v[l_idx, [7, 4, 1]]
        v[l_idx, [7, 4, 1]] = v[d_idx, [5, 4, 3]]
        v[d_idx, [5, 4, 3]] = v[r_idx, [1, 4, 7]]  # this one is special!
        v[r_idx, [1, 4, 7]] = tmp
        return self

    def rotate(self, face, clockwise=None, record=True):
        '''
        Operate the cube
        :param face: Which face has to be turned (clockwise or counterclockwise)
        :param clockwise: Whether it is clockwise
        :param record: Whether the rotation should be recorded into the _rec. The whole cube rotation should not be recorded.
        :return: None
        '''
        face = face.upper()

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
            self._try_push_record(face, record)
        else:
            self._c[f_idx] = (self._c[f_idx].T)[::-1, :]
            v[l_idx, l_nbr] = v[u_idx, u_nbr]
            v[u_idx, u_nbr] = v[r_idx, r_nbr]
            v[r_idx, r_nbr] = v[d_idx, d_nbr]
            v[d_idx, d_nbr] = temp
            self._try_push_record(face + "'", record)

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
                self._try_push_record("x", record)
            else:
                self.rotate_sequence("L'R", record=False)
                self._rotate_yz_middle()
                self._rotate_yz_middle()
                self._rotate_yz_middle()
                self._try_push_record("x'", record)
        elif axis[0] == 'y':
            if clockwise:
                self.rotate_sequence("FB'", record=False)
                self._rotate_xz_middle()
                self._try_push_record("y", record)
            else:
                self.rotate_sequence("F'B", record=False)
                self._rotate_xz_middle()
                self._rotate_xz_middle()
                self._rotate_xz_middle()
                self._try_push_record("y'", record)
        elif axis[0] == 'z':
            if clockwise:
                self.rotate_sequence("UD'", record=False)
                self._rotate_xy_middle()
                self._try_push_record("z", record)
            else:
                self.rotate_sequence("U'D", record=False)
                self._rotate_xy_middle()
                self._rotate_xy_middle()
                self._rotate_xy_middle()
                self._try_push_record("z'", record)
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
            raise ValueError("axis = {}, which has illegal letter".format(axis))
        return self

    def _rotate_or_view(self, op, record=True):
        op = op.upper()
        if op[0] in ["U", "D", "F", "B", "L", "R"]:
            self.rotate(op, record=record)
        else:
            self.view(op, record=record)
        return self

    def rotate_or_view_sequence(self, seq, record=True):
        '''
        This function can interpret a sequence of views or operations.
        If the sequence only contains operations but not views, it is recommended to use rotate_sequence function
        :param seq:
        :param record:
        :return:
        '''
        if len(seq) == 0:
            return
        if isinstance(seq, str):
            seq = self._parse_str_seq_to_list(seq)
        for r in seq:
            if r[-1] == '2':
                self._rotate_or_view(r[0], record=record)
                self._rotate_or_view(r[0], record=record)
            else:
                self._rotate_or_view(r, record=record)
        return self


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
        return cp['D'][0][0] == cp['D'][0][2] == cp['D'][2][0] == cp['D'][2][2] == cp['D'][1][1]

    def is_down_corner2_finished(self):
        cp = self.copy()
        cp.view("o", record=False)
        for i in range(4):
            if not (cp['F'][1][1] == cp['F'][2][0] == cp['F'][2][2] and cp['D'][0][0] == cp['D'][0][2] == cp['D'][1][1]):
                return False
            cp.view("z", record=False)
        return True

    def is_down_edge_finished(self):
        cp = self.copy()
        cp.view("o", record=False)
        for i in range(4):
            if not (cp['F'][2][1] == cp['F'][1][1] and cp['D'][0][1] == cp['D'][1][1]):
                return False
            cp.view("z", record=False)
        return True

    def is_face_solved(self, face):
        return np.all(self[face].reshape((-1,)) == self[face][1][1])

    def is_all_solved(self):
        return np.all([self.is_face_solved(f) for f in ['U', 'D', 'L', 'R', 'F', 'B']])


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
                cube._push_back_record(shortcut('L', i), label='up_cross/case_1/solution')
                cube.rotate_sequence("L'FL", record="up_cross/case_1/solution")
                cube.rotate_sequence(shortcut("L'", i), record='up_cross/case_1/solution')  # avoid messing up
                return True
            cube.rotate('L', record=False)   # For covering other cases
        return False

    def _case2(self, cube, desire):
        for i in range(4): # U on the right
            if cube['R'][2][1] == self._face_map['U'] and cube['D'][1][2] == desire:
                cube._push_back_record(shortcut('R', i), label='up_cross/case_2/solution')
                cube.rotate_sequence("RF'R'", record="up_cross/case_2/solution")
                cube.rotate_sequence(shortcut("R'", i), record='up_cross/case_2/solution')  # avoid messing up
                return True
            cube.rotate('R', record=False)   # For covering other cases
        return False

    def _case3(self, cube, desire):
        for i in range(4): # U on the front
            if cube['F'][2][1] == self._face_map['U'] and cube['D'][0][1] == desire:
                cube._push_back_record(shortcut('F', i), label='up_cross/case_3/solution')
                cube.rotate_sequence("DRF'R'", record='up_cross/case_3/solution')
                return True
            cube.rotate('F', record=False)   # For covering other cases
        return False

    def _case4(self, cube, desire):
        for i in range(4): # U on the back
            if cube['B'][2][1] == self._face_map['U'] and cube['D'][2][1] == desire:
                cube._push_back_record(shortcut('B', i), label='up_cross/case_4/solution')
                cube.rotate_sequence("D", record='up_cross/case_4/solution')
                cube.rotate_sequence(shortcut("B'", i), record="up_cross/case_4/solution")  # prevent from messing the others
                cube.rotate_sequence("L'FL", record="up_cross/case_4/solution")
                return True
            cube.rotate('B', record=False)   # For covering other cases
        return False

    def _case5(self, cube, desire):
        for i in range(4): # U on the down side
            if cube['D'][0][1] == self._face_map['U'] and cube['F'][2][1] == desire:
                cube._push_back_record(shortcut('D', i), label='up_cross/case_5/solution')
                cube.rotate_sequence("FF", record='up_cross/case_5/solution')
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
                cube._push_back_record(shortcut('D', i), label='up_corner/case_6/solution')
                cube.rotate_sequence("D'F'DF", record="up_corner/case_6/solution")
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
                cube._push_back_record(shortcut('D', i), label='up_corner/case_7/solution')
                cube.rotate_sequence("DLD'L'", record='up_corner/case_7/solution')
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
                cube._push_back_record(shortcut('D', i), label='up_corner/case_8/solution')
                cube.rotate_sequence("F'D'D'FD", record="up_corner/case_8/solution")
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
            cube.rotate_sequence("LDL'D'", record="up_corner/case_9/solution")
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
            cube.rotate_sequence("B'D'D'B", record='up_corner/case_10/solution')
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
            cube.rotate_sequence("L'DLD", record="up_corner/case_11/solution")
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
            cube.rotate_sequence("R'D'R", record="up_corner/case_12/solution")
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
            cube.rotate_sequence("R'D'R", record="up_corner/case_13_L/solution")
            return self._case8(cube, desire_f, desire_l)
        if cube['R'][0][0] == self._face_map['U'] and cube['F'][0][2] == desire_f and cube['U'][2][2] == desire_l: # 右面邻角的右面
            cube.rotate_sequence("R'D'R", record="up_corner/case_13_R/solution")
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
            cube.rotate_sequence("LDL'D'", record="up_corner/case_14/solution")
            return self._case8(cube, desire_f, desire_l)

    def _case15(self, cube, desire_f, desire_l):  # U在BR上排的情况
        if cube['B'][0][0] == self._face_map['U'] and cube['U'][0][2] == desire_l and cube['R'][0][2] == desire_f:
            cube.rotate_sequence("B'D'D'B", record="up_corner/case_15/solution")
            return self._case7(cube, desire_f, desire_l)
        if cube['R'][0][2] == self._face_map['U'] and cube['B'][0][0] == desire_l and cube['U'][0][2] == desire_f:
            cube.rotate_sequence("B'D'D'B", record="up_corner/case_15/solution")
            return self._case8(cube, desire_f, desire_l)

    def _case16(self, cube, desire_f, desire_l):  # U在BL上排的情况
        if cube['B'][0][2] == self._face_map['U'] and cube['L'][0][0] == desire_l and cube['U'][0][0] == desire_f:
            cube.rotate_sequence("BDB'", record="up_corner/case_16/solution")
            return self._case6(cube, desire_f, desire_l)
        if cube['L'][0][0] == self._face_map['U'] and cube['U'][0][0] == desire_l and cube['B'][0][2] == desire_f:
            cube.rotate_sequence("L'DDLD'", record="up_corner/case_16/solution")
            return self._case7(cube, desire_f, desire_l)

    # For the middle layer
    def _case17(self, cube, desire_f, desire_l):
        # 上换左
        for i in range(4):
            if cube['F'][0][1] == desire_f and cube['U'][2][1] == desire_l:
                cube._push_back_record(shortcut('U', i), label="middle_layer/case_17/solution")
                cube.rotate_sequence("U'L'ULUFU'F'", record="middle_layer/case_17/solution")  # The order of this formula is 15
                return True
            cube.rotate("U", record=False)
        return False

    def _case18(self, cube, desire_f, desire_l):
        # 上换右
        for i in range(4):
            if cube['U'][1][0] == desire_f and cube['L'][0][1] == desire_l:
                cube._push_back_record(shortcut('U', i), label="middle_layer/case_18/solution")
                cube.view("z'", record="middle_layer/case_18/solution")
                cube.rotate_sequence("URU'R'U'F'UF", record="middle_layer/case_18/solution")  # The order of this formula is 15
                cube.view("z", record="middle_layer/case_18/solution")
                return True
            cube.rotate("U", record=False)
        return False

    def _case19(self, cube, desire_f, desire_l):
        # 中层反位的情况
        if cube['L'][1][2] == desire_f and cube['F'][1][0] == desire_l:
            cube.view("z'", record="middle_layer/case_19/solution")
            cube.rotate_sequence("URU'R'U'F'UF", record="middle_layer/case_19/solution")  # We use formula in case 17 or 18
            cube.view("z", record="middle_layer/case_19/solution")
            if self._case17(cube, desire_f, desire_l):
                return True
            if self._case18(cube, desire_f, desire_l):
                return True
            raise NotImplementedError("Please check implementation")
        return False

    def _case19_1(self, cube):
        # 中层两个调换的情况
        if cube['F'][1][0] != cube['U'][1][1] and cube['L'][1][2] != cube['U'][1][1]:
            # 这时候找到一个上面或前面含有U颜色的
            for i in range(4):
                if (cube['F'][0][1] == cube['U'][1][1] or cube['U'][2][1] == cube['U'][1][1]) and not \
                        (cube['F'][1][0] == cube['F'][1][1] and cube['L'][1][2] == cube['L'][1][1]):
                    cube._push_back_record(shortcut('U', i), label="middle_layer/case_19_1/solution")
                    cube.rotate_sequence("U'L'ULUFU'F'", record="middle_layer/case_19_1/solution")  # 上换左公式来一个
                    return True
                cube.rotate("U", record=False)
        return False

    # For the down cross
    def _case20(self, cube):
        # 小拐角的情况
        for i in range(4):
            if cube['U'][1][0] == cube['U'][1][1] and cube['U'][0][1] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_cross/case_20/solution")
                cube.rotate_sequence("FURU'R'F'", record="down_cross/case_20/solution")
                return True
            cube.rotate("U", record=False)
        return False

    def _case21(self, cube):
        # 一字马的情况
        for i in range(4):
            if cube['U'][1][0] == cube['U'][1][1] and cube['U'][1][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_cross/case_21/solution")
                cube.rotate_sequence("FURU'R'F'", record="down_cross/case_21/solution")  # 规约为拐角情况
                return self._case20(cube)
            cube.rotate("U", record=False)
        return False

    def _case22(self, cube):
        # 一个黄点的情况，上面一定是黄色的，不用匹配
        cube.rotate_sequence("FURU'R'F'", record="down_cross/case_22/solution")
        return self._case21(cube)  # 规约为一字马的情况

    # For the down corner, see https://zhuanlan.zhihu.com/p/42299115
    def _case23(self, cube):  # 右手小鱼
        for i in range(4):
            if cube['U'][2][0] == cube['F'][0][2] == cube['B'][0][2] == cube['R'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_23/solution")
                cube.rotate_sequence("RUR'URUUR'", record='down_cross/case_23/solution')
                return True
            cube.rotate("U", record=False)
        return False

    def _case24(self, cube):  # 左手小鱼
        for i in range(4):
            if cube['U'][2][2] == cube['F'][0][0] == cube['B'][0][0] == cube['L'][0][0] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_24/solution")
                cube.rotate_sequence("L'U'LU'L'U'U'L", record="down_corner/case_24/solution")
                return True
            cube.rotate("U", record=False)
        return False

    def _case25(self, cube):  # 左上右下都有
        for i in range(4):
            if cube['U'][0][0] == cube['U'][2][2] == cube['F'][0][0] == cube['R'][2][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_25/solution")
                cube.rotate_sequence("RUR'URUUR'", record="down_corner/case_25/solution")
                return self._case24(cube)
            if cube['U'][2][0] == cube['U'][0][2] == cube['L'][0][0] == cube['F'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_25/solution")
                cube.rotate_sequence("L'U'LU'L'U'U'L", record="down_corner/case_25/solution")
                return self._case23(cube)
            cube.rotate("U", record=False)
        return False

    def _case26(self, cube):  # 左上右上
        for i in range(4):
            if cube['U'][0][0] == cube['F'][0][0] == cube['F'][0][2] == cube['U'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_26/solution")
                cube.rotate_sequence("RUR'URUUR'", record="down_corner/case_26/solution")
                return self._case24(cube)
            cube.rotate("U", record=False)
        return False

    def _case27(self, cube):  # 右上右下
        for i in range(4):
            if cube['F'][0][0] == cube['U'][0][2] == cube['U'][2][2] == cube['B'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_27/solution")
                cube.rotate_sequence("RUR'URUUR'", record="down_corner/case_27/solution")
                return self._case24(cube)
            cube.rotate("U", record=False)
        return False

    def _case28(self, cube):  # 只有十字1
        for i in range(4):
            if cube['L'][0][2] == cube['R'][0][0] == cube['R'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_28/solution")
                cube.rotate_sequence("RUR'URUUR'", record="down_corner/case_28/solution")
                return self._case23(cube)
            cube.rotate("U", record=False)
        return False

    def _case29(self, cube):  # 只有十字2
        for i in range(4):
            if cube['L'][0][2] == cube['F'][0][2] == cube['U'][1][1]:
                cube._push_back_record(shortcut('U', i), label="down_corner/case_29/solution")
                cube.rotate_sequence("RUR'URUUR'", record="down_corner/case_29/solution")
                return self._case23(cube)
            cube.rotate("U", record=False)
        return False


    # For the down corner of other faces, https://zhuanlan.zhihu.com/p/42331476
    def _case30(self, cube):
        for i in range(4):
            if cube['L'][0][0] == cube['L'][0][2] == cube['L'][1][1]:
                cube.view("z'")
            if cube['R'][0][0] == cube['R'][0][2] == cube['R'][1][1]:
                cube.view("z")
            if cube['B'][0][0] == cube['B'][0][2] == cube['B'][1][1]:
                cube.view("z")
                cube.view("z")
            if cube['F'][0][0] == cube['F'][0][2] == cube['F'][1][1]:  # Above three conditions will fall through to here
                cube._push_back_record(shortcut('U', i), label="down_corner_2/case_30/solution")
                cube.view("z'", record="down_corner_2/case_30/solution")
                cube.view("y'", record="down_corner_2/case_30/solution")
                cube.rotate_sequence("UURU'U'R'FF", record="down_corner_2/case_30/solution")
                cube.rotate_sequence("U'U'L'UULF'F'", record="down_corner_2/case_30/solution")
                cube.view("y", record="down_corner_2/case_30/solution")
                cube.view("z", record="down_corner_2/case_30/solution")
                for j in range(4):
                    if not (cube['F'][0][0] == cube['F'][2][2] == cube['F'][1][1]):
                        cube.rotate("U", record=False)
                    else:
                        cube._push_back_record(shortcut('U', j), label="down_corner_2/case_30/solution")
                        return True
                print(cube)
                raise NotImplementedError
            cube.rotate("U", record=False)
        return False

    # For the last step of solving the down edge
    def _case31(self, cube):
        if cube['L'][0][1] == cube['R'][1][1] and cube['R'][0][1] == cube['F'][1][1] and cube['F'][0][1] == cube['L'][1][1]:
            cube.rotate_sequence("RUR'URUUR'", record="down_edge/case_31/solution")
            cube.view("z", record="down_edge/case_31/solution")
            cube.rotate_sequence("L'U'LU'L'U'U'L", record="down_edge/case_31/solution")
            return True
        return False

    def _case32(self, cube):
        if cube['L'][0][1] == cube['F'][1][1] and cube['F'][0][1] == cube['R'][1][1] and cube['R'][0][1] == cube['L'][1][1]:
            cube.rotate_sequence("L'U'LU'L'U'U'L", record="down_edge/case_31/solution")
            cube.view("z'", record="down_edge/case_31/solution")
            cube.rotate_sequence("RUR'URUUR'", record="down_edge/case_31/solution")
            return True
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
            cube.rotate_sequence("U'", record="up_cross/case_5_1/solution")
            return
        if cube['U'][1][2] == self._face_map['U'] and cube['R'][0][1] == desire:
            cube.rotate_sequence("R'", record="up_cross/case_5_2/solution")
            if self._case3(cube, desire):
                return
        if cube['U'][0][1] == self._face_map['U'] and cube['B'][0][1] == desire:
            cube.rotate_sequence("B'", record="up_cross/case_5_3/solution")
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
        if cube.is_up_cross_finished():
            return
        for i in range(4):
            self.up_cross_one(cube, cube['F'][1][1], i)
            cube.view('z', record='up_cross/transition')
        return

    def solve_up_corner(self, cube):
        '''
        这里的思路是还原ULF，然后转魔方
        :param cube:
        :return:
        '''
        if cube.is_up_corner_finished():
            return
        for i in range(4):
            desire_l = cube['L'][1][1]
            desire_f = cube['F'][1][1]
            self.up_corner_one(cube, desire_f, desire_l)
            cube.view('z', record='up_corner/transition')

    def solve_middle_layer(self, cube):
        cube.view("x", record="middle_layer/transition")
        cube.view("x", record="middle_layer/transition")
        fail_cnt = 0    # If no operation happens after 4 rotations, we will know that some conditions are not covered
        while not cube.is_middle_layer_finished():
            if not (cube['L'][1][2] == cube['L'][1][1] and cube['F'][1][0] == cube['F'][1][1]):
                if self._case17(cube, cube['F'][1][1], cube['L'][1][1]):
                    fail_cnt = 0
                elif self._case18(cube, cube['F'][1][1], cube['L'][1][1]):
                    fail_cnt = 0
                elif self._case19(cube, cube['F'][1][1], cube['L'][1][1]):
                    fail_cnt = 0
                else:
                    fail_cnt += 1
            if fail_cnt == 5:
                cube.pop_back_record()   # Pop four 'z' rotations
                cube.pop_back_record()
                cube.pop_back_record()
                cube.pop_back_record()
                if self._case19_1(cube):
                    fail_cnt = 0
                else:
                    print(cube)
                    raise NotImplementedError("Some condition may not be implemented in solving middle layer")
            else:
                cube.view("z", record="middle_layer/transition")

    def solve_down_cross(self, cube):
        if cube.is_down_cross_finished():
            return
        if self._case20(cube):
            return
        if self._case21(cube):
            return
        if self._case22(cube):
            return

    def solve_down_corner(self, cube):
        if cube.is_down_corner_finished():
            return
        if self._case23(cube):
            return
        if self._case24(cube):
            return
        if self._case25(cube):
            return
        if self._case26(cube):
            return
        if self._case27(cube):
            return
        if self._case28(cube):
            return
        if self._case29(cube):
            return
        return

    def solve_down_corner_2(self, cube):
        fail_cnt = 0
        while not cube.is_down_corner2_finished():
            if self._case30(cube):
                if cube.is_down_corner2_finished():
                    return
            else:
                fail_cnt += 1
                if fail_cnt > 3:
                    raise ValueError("Too much failing in solving down corner. Optimize.")
                cube.view("z'", record="down_corner_2/transition")
                cube.view("y'", record="down_corner_2/transition")
                cube.rotate_sequence("UURU'U'R'FF", record="down_corner_2/transition")
                cube.rotate_sequence("U'U'L'UULF'F'", record="down_corner_2/transition")
                cube.view("y", record="down_corner_2/transition")
                cube.view("z", record="down_corner_2/transition")

    def solve_down_edge(self, cube):

        def try_down_edge():
            if cube.is_down_edge_finished():
                return True
            if cube.is_face_solved('L'):
                cube.view("z", record="down_corner_2/transition")
            if cube.is_face_solved('R'):
                cube.view("z'", record="down_corner_2/transition")
            if cube.is_face_solved('F'):
                cube.view("z", record="down_corner_2/transition")
                cube.view("z", record="down_corner_2/transition")
            if cube.is_face_solved('B'):
                if self._case31(cube):
                    return True
                if self._case32(cube):
                    return True
                raise NotImplementedError
            cube.rotate_sequence("RUR'URUUR'", record="down_edge/case_33/solution")
            cube.view("z", record="down_edge/case_33/solution")
            cube.rotate_sequence("L'U'LU'L'U'U'L", record="down_edge/case_33/solution")
            return False

        fail_cnt = 0
        while True:
            try_down_edge()
            if cube.is_all_solved():
                break
            else:
                fail_cnt += 1
                if fail_cnt > 5:
                    raise ValueError("Fail time is too much. Optimize")
        cube.view('o', record=False)


    def solve(self, cube: Cube, inplace=False, steps=99):
        if not inplace:
            cube = cube.copy()
        cube.clear_record()

        if steps >= 1:
            self.solve_up_cross(cube)
        if steps >= 2:
            self.solve_up_corner(cube)
        if steps >= 3:
            self.solve_middle_layer(cube)
        if steps >= 4:
            self.solve_down_cross(cube)
        if steps >= 5:
            self.solve_down_corner(cube)
        if steps >= 6:
            self.solve_down_corner_2(cube)
        if steps >= 7:
            self.solve_down_edge(cube)
        return cube


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
def create_random_cube(return_op=False):
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


def _create_step_ut(samples, seed, until_step, f_conditions):

    np.random.seed(seed)
    def _ut_fn():
        T = samples
        for i in trange(T):
            cube, steps = create_random_cube(return_op=True)
            cube_c = cube.copy()   # The cube for turning
            cube_f = cube.copy()   # The cube for applying the formula
            solver = LBLSolver()

            try:
                solver.solve(cube_c, inplace=True, steps=until_step)
                rec = cube_c.get_record()
                cube_f.rotate_or_view_sequence(rec, record=False)
                for cond in f_conditions:
                    if not cond(cube_c):
                        raise ValueError("The cube is not solved as expected")
                    if not cond(cube_f):
                        raise ValueError("The record is not worked as expected though cube is solved")

            except ValueError as e:
                print("=======Error Happens========")
                print(str(e))
                print("=======Original cube========")
                print(cube)
                print("=======See clearly of solving cube==========")
                for k in range(4):
                    print(cube_c)
                    cube_c.view("z", record=False)
                    print("===========")
                print("=======Cloned cube======")
                print(cube_f)
                print("=======Rotate sequence======")
                print(" ".join(steps))
                print(" ".join(cube_c.get_record()))
                break
    return _ut_fn()

def _ut_up_cross():
    _create_step_ut(1000, 42, 1, [Cube.is_up_cross_finished])

def _ut_up_corner():
    _create_step_ut(1000, 42, 2, [Cube.is_up_cross_finished, Cube.is_up_corner_finished])


def _ut_middle_layer():
    _create_step_ut(1000, 42, 3, [Cube.is_up_cross_finished, Cube.is_up_corner_finished, Cube.is_middle_layer_finished])


def _ut_bottom_cross():
    _create_step_ut(1000, 43, 4, [Cube.is_up_cross_finished, Cube.is_up_corner_finished, Cube.is_middle_layer_finished,
                                  Cube.is_down_cross_finished])

def _ut_bottom_corner():
    _create_step_ut(1000, 42, 5, [Cube.is_up_cross_finished, Cube.is_up_corner_finished, Cube.is_middle_layer_finished,
                                  Cube.is_down_cross_finished, Cube.is_down_corner_finished])

def _ut_bottom_corner2():
    _create_step_ut(1000, 42, 6, [Cube.is_up_cross_finished, Cube.is_up_corner_finished, Cube.is_middle_layer_finished,
                                  Cube.is_down_cross_finished, Cube.is_down_corner_finished,
                                  Cube.is_down_corner2_finished])
def _ut_all_solved():
    _create_step_ut(100000, 42, 7, [Cube.is_all_solved])

def _ut_basic_op():
    np.random.seed(43)
    cube = create_random_cube()
    print(cube)
    cube.view("y'")
    print(cube)

def one_demo():
    np.random.seed(20)
    cube, ops = create_random_cube(return_op=True)
    # cube = Cube()
    # cube.rotate_sequence("LL", record=False)
    solver = LBLSolver()
    solver.solve(cube)
    print("===========Original cube===============")
    print(cube)
    print("===========Sequence===========")
    print(" ".join(ops))
    stages = [solver.solve_up_cross, solver.solve_up_corner, solver.solve_middle_layer,
              solver.solve_down_cross, solver.solve_down_corner, solver.solve_down_corner_2, solver.solve_down_edge]
    for i in range(len(stages)):
        stages[i](cube)
        print("===============Stage {} finished. {} steps.===============".format(i + 1, len(cube.get_record())))
        print(cube)
    if cube.is_all_solved():
        print("===============All done! The answer is: ==================")
        print(" ".join(cube.get_record()))
        print(cube.get_op_label())
    else:
        print("==============Whoops!! Please Check!!!!!================")

def _ut_mirror():
    np.random.seed(42)
    T = 10000
    for t in trange(T):
        cube1 = create_random_cube()
        cube2 = cube1.copy()
        solver = LBLSolver()
        solver.solve(cube1, inplace=True)
        rec = cube1.get_mirroring_record()
        for r in rec:
            if r[0] not in ['x', 'y', 'z']:
                cube2.rotate(r, record=False)
        if cube1 != cube2:
            print("=======Error happens========")
            print(cube1)
            print("=======Cloned cube==========")
            print(cube2)
            print("=======Result===============")
            print(" ".join(rec))
            raise ValueError("Test mirror implementation error")


def regression_test():
    # _ut_basic_op()
    # _ut_up_cross()
    # _ut_up_corner()
    # _ut_middle_layer()
    # _ut_bottom_cross()
    # _ut_bottom_corner()
    # _ut_bottom_corner2()
    # _ut_all_solved()
    _ut_mirror()

if __name__ == '__main__':
    regression_test()
    # one_demo()