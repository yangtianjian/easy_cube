import web
import json
from cube_solver.solver import LBLSolver, KociembaSolver, Cube, create_random_cube
from colordetect.cnn_detector import CNNModel, ConvolutionalDetector, Lenet5Module
import numpy as np
import functools
import random
from tqdm import trange
import os
import pickle
import sys, logging
from wsgilog import WsgiLog

class Log(WsgiLog):
    def __init__(self, application):
        WsgiLog.__init__(
            self,
            application,
            logformat = '%(message)s',
            tofile = True,
            toprint = True,
            file = './server.log',
            interval='H')

urls = (
    '/solve_lbl', 'SolveLBL',
    '/solve_koci', 'SolveKoci',
    '/recognize', 'RecognizeColor',
    '/generate', 'Generate',
    '/', "Test"
)

color_detector = CNNModel('./colordetect/models/best_model')
_color_map = {
        'Y': 0,
        'B': 1,
        'O': 2,
        'W': 3,
        'G': 4,
        'R': 5
    }

program_mapping = {
    1: "up_cross",
    2: "up_corner",
    3: "middle_layer",
    4: "down_cross",
    5: "down_corner",
    6: "down_corner_2",
    7: "down_edge"
}

problem_cache = {}  # cache[x][y][z]: the z-th sample of the y-th case in program x

random.seed(40)
np.random.seed(40)

def returns_json(func):
    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        # web.header("Content-Type", "application/json;charset=UTF-8")
        data = json.loads(web.data())
        if 'magic_number' not in data or data['magic_number'] != 'qwertyuiop':
            return {
                "success": False,
                "message": "Please add magic number. Or check if magic number is correct."
            }
        else:
            return json.dumps(func(obj, data, *args, **kwargs))
    return wrapper


def get_case_apply_interval(program_number, case_number, op_label):

    head = None
    tail = None

    l = len(op_label)
    for i in range(l):
        start, end, op_name = op_label[i]
        program, case_text, stage = op_name.split("/")
        case_num = case_text.split("_")[1]
        if i == l - 1 and head is not None:
            tail = end + 1
            break
        elif head is None and program_mapping[program_number] == program and str(case_number) == case_num and stage == "solution":
            head = start
            if i == l - 1:
                tail = end + 1
                break
        elif head is not None:
            if stage == "next":
                tail = start
                break
            elif program_number < len(program_mapping):
                if program_mapping[program_number + 1] == program:
                    tail = start
                    break
            else:
                if stage != "solution":
                    tail = start
                    break
    return (head, tail)


def create_cube(program_number, case_number):
    fail_cnt = 0
    while True:
        cube, _ = create_random_cube(return_op=True)
        cube_cp = cube.copy()
        LBLSolver().solve(cube, inplace=True, steps=program_number)

        cube_rec = cube.get_record()
        cube_op_lbl = cube.get_op_label()
        cube_mirror_rec = cube.get_mirroring_record()

        head, tail = get_case_apply_interval(program_number, case_number, cube_op_lbl)
        if head is None or tail is None:
            fail_cnt += 1
            continue
        if fail_cnt == 100:
            print("program {} case {} too many trials".format(program_number, case_number))
            exit(0)

        question_cube = cube_cp.copy().rotate_sequence(cube_mirror_rec[: head], record=False, mask_whole_rotations=True)
        question_cube_2 = cube_cp.copy().rotate_or_view_sequence(cube_rec[: head], record=False)

        # print(question_cube_2)

        steps_to_achieve = KociembaSolver(terminal=question_cube).solve(Cube())
        steps_to_achieve += [x for x in cube_rec[: head] if x[0] in ['x', 'y', 'z']]

        assert Cube().rotate_or_view_sequence(steps_to_achieve, record=False) == question_cube_2

        return {
            "cube": question_cube_2.to_json_object(),
            "finish_cube": cube.to_json_object(),
            "trans": steps_to_achieve,
            "answer": cube_rec[head: tail],
            "answer_mirror": cube_mirror_rec[head: tail]
        }


def create_problem_cache():
    cache_ = {}
    cases = [0, 0, 0, 3, 3, 7, 2, 3]
    cache_volume = 10
    for x in trange(3, 8):
        cache_[x] = {}
        for y in range(1, cases[x] + 1):
            cache_[x][y] = [None] * cache_volume
            for t in range(cache_volume):
                cache_[x][y][t] = create_cube(x, y)
    return cache_

def read_or_create_cache(path, create_fn):
    if not os.path.exists(path):
        cache_ = create_fn()
        with open(path, "wb") as f:
            pickle.dump(cache_, f)
        return cache_
    else:
        with open(path, "rb") as f:
            cache_ = pickle.load(f)
        return cache_


class SolveLBL():


    @returns_json
    def POST(self, data):
        '''
        0: yellow
        1: blue
        2: orange
        3: white
        4: green
        5: red

        API conventions:
        cube: {
            "F": [],
            "U": [],
            "D": [],
            "B": [],
            "L": [],
            "R": []
        }
        :return
        {
            success: True,
            message: "",
            solution: [],
            solution_with_mirror: [],
            solution_span: [
            [1, 3, "First layer"],
            ...
            ]
        }
        '''
        try:
            cube = Cube.from_json(data['cube'])
            solver = LBLSolver()
            solver.solve(cube, inplace=True)
            solution = cube.get_record()
            solution_with_mirror = cube.get_mirroring_record()
            solution_span = [list(x) for x in cube.get_op_label()]
            return {
                "success": True,
                "message": "Success",
                "solution": solution,
                "solution_with_mirror": solution_with_mirror,
                "solution_span": solution_span
            }
        except ValueError as e:
            return {
                "success": False,
                "message": "Cube is not valid.",
                "solution": [],
                "solution_with_mirror": [],
                "solution_span": []
            }

class SolveKoci():
    '''
        API conventions:
        cube: {
            "F": [],
            "U": [],
            "D": [],
            "B": [],
            "L": [],
            "R": []
        }

        terminal: {
            "F": [],
            "U": [],
            "D": [],
            "B": [],
            "L": [],
            "R": []
        }

        :return: {
            success: True,
            message: "",
            solution: []
        }
    '''
    @returns_json
    def POST(self, data):
        try:
            cube = Cube.from_json(data['cube'])

            terminal = Cube()
            if 'terminal' in data:
                terminal = Cube.from_json(data['terminal'])
            solver = KociembaSolver(terminal=terminal)
            ans = solver.solve(cube)
            return {
                "success": True,
                "message": "Success",
                "solution": ans
            }
        except ValueError as e:
            return {
                "success": False,
                "message": "Cube is not valid.",
                "solution": []
            }

class RecognizeColor():

    @returns_json
    def POST(self, data):
        '''
        picture : {
            width: 1024,
            height: 768
            U: {
                R: [],
                G: [],
                B: [],

            },
            D: {
                R: [],
                G: [],
                B: [],
            }

            ...
        }
        :return: {
            success: True,
            message: ""
            cube: {
                U: [],
                L: [],
                ...
            },
            trans: []
        }
        '''
        face_colors = {}
        for face in ['F', 'L', 'D', 'U', 'R', 'B']:
            img_data = data['picture'][face]
            img_arr = []
            for k in ['B', 'G', 'R']:
                channel_data = np.array(img_data[k])
                channel_data = channel_data.reshape(data['picture']['height'], data['picture']['width'])
                img_arr.append(channel_data)
            img_arr = np.stack(img_arr, axis=-1)
            colors = color_detector.predict(img_arr)
            face_colors[face] = [_color_map[x] for x in colors]
        try:
            cube = Cube.from_json(face_colors, check_valid=True)
            formula = KociembaSolver(terminal=cube).solve(Cube())
            return {
                "success": True,
                "message": "Success",
                "cube": face_colors,
                "trans": formula
            }
        except ValueError as e:
            return {
                "success": False,
                "message": "The cube is invalid. Please check the cube.",
                "cube": face_colors,
                "trans": []
            }

class Generate():
    '''
    生成一个魔方"问题"。对于公式专项练习，随机种子固定
    目前有这几个公式的教学：
    1. 右手中棱归位
    2. 左手中棱归位
    3. 底层十字
    4. 底面复原
    5. 底棱复原（最终复原）
    {
        "program": 1,    # 第几步的专项练习
        "case": 1,       # 第几种情况的专项练习
        "type": "train" / "test"   # 练习还是测试
    }

    :return {
        "success": True , # 如果是False, 有可能是前端给的情况编号出现错误
        "message": "success"
        "cube": {
            "U": [],
            "D": []
            ...
        },
        "trans": []       # 如何从原始状态达到这一步
        "answer": []      # 从这一步如何转完成公式
        "finish_cube": {  #
            "U": [],
            "D": [],
            ...
        }
    }

    前端需要在下面几种情况开放训练 / 测试
    program 3: middle layer(中层还原). case 1 (上换左) & case 2（上换右） & case 3（中间调换）
    program 4: down cross(底面十字构造). case 1: （小拐角） & case 2(一字马) & case 3（单个中心点)
    program 5: down corner(底角的底面色正确还原) case 1: (右手小鱼) case 2: 左手小鱼 case 3: Z字 case 4: "由"字1， case 5: "由"字2，case 6: 十字1， case 7： 十字2
    program 6: down corner2(底角的在正确的位置) case 1: 能找到小船  case 2: 不能找到小船
    program 7: down edge(完成魔方复原) case 1: 逆时针  case 2: 顺时针  case 3: L, R, D, B 都没有复原好
    '''


    @returns_json
    def POST(self, data):
        try:
            program, case, typ = int(data['program']), int(data['case']), data['type']
            if program not in problem_cache:
                return {
                    "success": False,
                    "message": "No such program: Program: {}".format(program)
                }
            if case not in problem_cache[program] or len(problem_cache[program][case]) == 0:
                return {
                    "success": False,
                    "message": "No such case, case {} in program {}".format(case, program)
                }
            typ = typ.lower()
            if typ == "test" or typ == "train":
                if typ == "train":
                    cube = problem_cache[program][case][0]
                else:
                    cube = random.choice(problem_cache[program][case])
                return dict({
                    "success": True,
                    "message": "Success"
                }, **cube)
            else:
                raise ValueError("Invalid type, type {} in the argument".format(typ))
        except ValueError as e:
            return {
                "success": False,
                "message": str(e)
            }

class Test():
    def GET(self):
        return "<html><head></head><body> <h1>It works !</h1></body></html>"


problem_cache = read_or_create_cache('./cube_solver/problems/problem_cache.pkl', create_problem_cache)
if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run(Log)