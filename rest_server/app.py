import web
import json
from cube_solver.solver import LBLSolver, KociembaSolver, Cube

urls = (
    '/solve_lbl', 'SolveLBL',
    '/solve_koci', 'SolveKoci'
    # '/recognize', 'RecognizeColor'
)

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

# class RecognizeColor():
#     pass


if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()