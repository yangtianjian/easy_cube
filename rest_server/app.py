import web
import json
from cube_solver.solver import LBLSolver, KociembaSolver, Cube
from colordetect.cnn_detector import CNNModel
import numpy as np

urls = (
    '/solve_lbl', 'SolveLBL',
    '/solve_koci', 'SolveKoci',
    '/recognize', 'RecognizeColor',
    '/teach', 'Teach',
    '/exam', 'Exam'
)

color_detector = CNNModel()
_color_map = {
        'Y': 0,
        'B': 1,
        'O': 2,
        'W': 3,
        'G': 4,
        'R': 5
    }

class SolveLBL():


    def POST(self):
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
        web.header("Content-Type", "application/json;charset=UTF-8")
        data = json.loads(web.data())

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

    def POST(self):
        web.header("Content-Type", "application/json;charset=UTF-8")
        data = json.loads(web.data())

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

    def POST(self):
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
        web.header("Content-Type", "application/json;charset=UTF-8")
        data = json.loads(web.data())

        face_colors = {}
        for face in ['F', 'L', 'D', 'U', 'R', 'B']:
            img_data = data['picture'][face]
            img_arr = []
            for k in ['B', 'G', 'R']:
                channel_data = np.array(img_data['k'])
                channel_data = channel_data.reshape(data['picture']['height'], data['picture']['width'])
                img_arr.append(channel_data)
            img_arr = np.stack(img_arr, axis=0)
            colors = color_detector.predict(img_arr)
            face_colors[face] = [_color_map[x] for x in colors]
        cube = Cube.from_json(face_colors)
        try:
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
                "cube": {},
                "trans": []
            }


if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()