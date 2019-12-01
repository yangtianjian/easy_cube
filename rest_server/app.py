import web
import json
from cube_solver.solver import LBLSolver, KociembaSolver, Cube

urls = (
    '/solve_lbl', 'SolveLBL',
    '/solve_koci', 'SolveKoci'
    '/recognize', 'RecognizeColor'
)

class SolveLBL():

    def POST(self):
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
        :return: {
            success: True,
            message: "",
            solution: [],
            solution_with_mirroring: [],
            solution_span: [
            [1, 3, "First layer"],
            ...
            ]
        }
        '''
        web.header("Content-Type", "application/json;charset=UTF-8")
        data = web.data()

        try:
            cube = Cube.from_json(data)
            solution = cube.get_record()


        except ValueError as e:
            return {
                "success": False,
                "message": "Cube is not valid."
            }



class SolveKoci():
    pass


class RecognizeColor():
    pass


if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()