/*cppimport
<%
cfg['include_dirs'] = ['/Users/apple/.local/include/python3.6m', '/Users/apple/anaconda3/envs/deep_learning_36/include/python3.6m']
cfg['compiler_args'] = ['-v', '-stdlib=libc++', '-mmacosx-version-min=10.9', '-std=gnu++11']
setup_pybind11(cfg)
%>
*/

#define DEBUG

#ifndef DEBUG
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iostream>

using namespace std;

//-----------------------------------------------
// We make up face on the top and front face on the front
struct Cube_t {
    char up[9];
    char front[9];
    char back[9];
    char ri[9];
    char lft[9];
    char down[9];
    char* cb[6] = {up, front, back, ri, lft, down};

    Cube_t() {
        for(int i = 0; i < 6; i++) {
            memset(cb[i], 0, 9 * sizeof(char));
        }
    }

    string to_string() {
        string ret;
        for(int i = 0; i < 6; i++) {
            for(int j = 0; j < 9; j++) {
                ret += cb[i][j];
            }
        }
        return ret;
    }
};

struct Record_t {
    vector<string> moves;

    string to_string() {
        string ret;
        for(int i = 0; i < moves.size(); i++) ret += moves[i];
        return ret;
    }
};

//----------------------------------
//static void display(char face[9]) {
//    for (int i = 0; i < 9; i++) {
//        cout << face[i] << " ";
//    }
//    cout << endl << endl;
//}

static void swap(char &a, char &b) {
    char t = a;
    a = b;
    b = t;
}

static void operate_clock(Cube_t& cb, char choice) {
    if (choice == 'w') {
        swap(cb.up[7], cb.up[3]);
        swap(cb.up[6], cb.up[4]);
        swap(cb.up[0], cb.up[2]);
        swap(cb.up[7], cb.up[5]);
        swap(cb.up[0], cb.up[4]);
        swap(cb.up[1], cb.up[3]);
        //-------------------------
        swap(cb.ri[0], cb.back[0]);
        swap(cb.ri[7], cb.back[7]);
        swap(cb.ri[6], cb.back[6]);
        swap(cb.back[6], cb.lft[6]);
        swap(cb.back[7], cb.lft[7]);
        swap(cb.back[0], cb.lft[0]);
        swap(cb.lft[6], cb.front[6]);
        swap(cb.lft[7], cb.front[7]);
        swap(cb.lft[0], cb.front[0]);
    }
        //-------------------------
    else if (choice == 'r') {
        swap(cb.front[0], cb.front[4]);
        swap(cb.front[7], cb.front[5]);
        swap(cb.front[1], cb.front[3]);
        swap(cb.front[0], cb.front[6]);
        swap(cb.front[1], cb.front[5]);
        swap(cb.front[2], cb.front[4]);
        //-------------------------
        swap(cb.ri[6], cb.up[3]);
        swap(cb.ri[5], cb.up[2]);
        swap(cb.ri[4], cb.up[1]);
        swap(cb.up[1], cb.lft[0]);
        swap(cb.up[2], cb.lft[1]);
        swap(cb.up[3], cb.lft[2]);
        swap(cb.lft[0], cb.down[3]);
        swap(cb.lft[1], cb.down[2]);
        swap(cb.lft[2], cb.down[1]);
        //--------------------------
    }
        //-----------------------------------
    else if (choice == 'y') {
        swap(cb.down[1], cb.down[5]);
        swap(cb.down[2], cb.down[4]);
        swap(cb.down[0], cb.down[6]);
        swap(cb.down[1], cb.down[3]);
        swap(cb.down[0], cb.down[4]);
        swap(cb.down[7], cb.down[5]);
        //--------------------------
        swap(cb.ri[4], cb.front[4]);
        swap(cb.ri[3], cb.front[3]);
        swap(cb.ri[2], cb.front[2]);
        swap(cb.front[2], cb.lft[2]);
        swap(cb.front[3], cb.lft[3]);
        swap(cb.front[4], cb.lft[4]);
        swap(cb.lft[4], cb.back[4]);
        swap(cb.lft[3], cb.back[3]);
        swap(cb.lft[2], cb.back[2]);
        //--------------------------
    }
        //-------------------------------------
    else if (choice == 'o') {
        swap(cb.back[4], cb.back[0]);
        swap(cb.back[3], cb.back[1]);
        swap(cb.back[5], cb.back[7]);
        swap(cb.back[4], cb.back[2]);
        swap(cb.back[5], cb.back[1]);
        swap(cb.back[6], cb.back[0]);
        //--------------------------
        swap(cb.ri[2], cb.down[5]);
        swap(cb.ri[1], cb.down[6]);
        swap(cb.ri[0], cb.down[7]);
        swap(cb.down[5], cb.lft[6]);
        swap(cb.down[6], cb.lft[5]);
        swap(cb.down[7], cb.lft[4]);
        swap(cb.lft[6], cb.up[7]);
        swap(cb.lft[5], cb.up[6]);
        swap(cb.lft[4], cb.up[5]);
        //--------------------------
    }
        //-------------------------------------
    else if (choice == 'g') {
        swap(cb.lft[6], cb.lft[2]);
        swap(cb.lft[5], cb.lft[3]);
        swap(cb.lft[7], cb.lft[1]);
        swap(cb.lft[4], cb.lft[6]);
        swap(cb.lft[7], cb.lft[3]);
        swap(cb.lft[0], cb.lft[2]);
        //--------------------------
        swap(cb.up[5], cb.back[2]);
        swap(cb.up[4], cb.back[1]);
        swap(cb.up[3], cb.back[0]);
        swap(cb.down[3], cb.back[2]);
        swap(cb.down[4], cb.back[1]);
        swap(cb.down[5], cb.back[0]);
        swap(cb.down[3], cb.front[6]);
        swap(cb.down[4], cb.front[5]);
        swap(cb.down[5], cb.front[4]);
        //--------------------------
    }
        //-------------------------------------------
    else if (choice == 'b') {
        swap(cb.ri[1], cb.ri[7]);
        swap(cb.ri[2], cb.ri[6]);
        swap(cb.ri[5], cb.ri[3]);
        swap(cb.ri[2], cb.ri[0]);
        swap(cb.ri[7], cb.ri[3]);
        swap(cb.ri[6], cb.ri[4]);
        //--------------------------
        swap(cb.down[1], cb.back[4]);
        swap(cb.down[0], cb.back[5]);
        swap(cb.down[7], cb.back[6]);
        swap(cb.up[7], cb.back[4]);
        swap(cb.up[0], cb.back[5]);
        swap(cb.up[1], cb.back[6]);
        swap(cb.up[7], cb.front[0]);
        swap(cb.up[0], cb.front[1]);
        swap(cb.up[1], cb.front[2]);
        //--------------------------
    }
}

static void rotate_clock(Record_t& rec, Cube_t& c, char choice) {
    rec.moves.push_back(string(1, choice));
    operate_clock(c, choice);
}

static void rotate_anticlock(Record_t& rec, Cube_t& c, char choice) {
    rec.moves.push_back(string(1, choice) + "'");
    operate_clock(c, choice);
    operate_clock(c, choice);
    operate_clock(c, choice);
}

static void white_bottom(Record_t& rec, Cube_t& cb, char q) {
    if ((cb.down[0] == 'w' && cb.ri[3] == q) || (cb.down[2] == 'w' && cb.front[3] == q) ||
        (cb.down[4] == 'w' && cb.lft[3] == q) || (cb.down[6] == 'w' && cb.back[3] == q)) {
        if (q == 'b') {
            while (cb.ri[3] != q || cb.down[0] != 'w') { rotate_clock(rec, cb, 'y'); }
        }
        if (q == 'r') {
            while (cb.front[3] != q || cb.down[2] != 'w') { rotate_clock(rec, cb, 'y'); }
            if (q != 'b') {
                while (cb.up[0] != 'w' && cb.ri[7] != 'b') { rotate_clock(rec, cb, 'w'); }
            }
        }
        if (q == 'g') {
            while (cb.lft[3] != q || cb.down[4] != 'w') { rotate_clock(rec, cb, 'y'); }
            if (q != 'b') {
                while (cb.up[0] != 'w' && cb.ri[7] != 'b') { rotate_clock(rec, cb, 'w'); }
            }
        }
        if (q == 'o') {
            while (cb.back[3] != q || cb.down[6] != 'w') { rotate_clock(rec, cb, 'y'); }
            if (q != 'b') {
                while (cb.up[0] != 'w' && cb.ri[7] != 'b') { rotate_clock(rec, cb, 'w'); }
            }
        }
        rotate_clock(rec, cb, q);
        rotate_clock(rec, cb, q);
    }
}

static void right_alg(Record_t& rec, Cube_t& cb, char a, char c) {
    rotate_anticlock(rec, cb, a);
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, a);
    white_bottom(rec, cb, c);
}

static void white_right(Record_t& rec, Cube_t& cb, char q) {
    if (cb.ri[1] == 'w' || cb.front[1] == 'w' || cb.lft[1] == 'w' || cb.back[1] == 'w') {
        if (cb.ri[5] == q && cb.front[1] == 'w') { right_alg(rec, cb, 'b', q); }
        if (cb.front[5] == q && cb.lft[1] == 'w') { right_alg(rec, cb, 'r', q); }
        if (cb.lft[5] == q && cb.back[1] == 'w') { right_alg(rec, cb, 'g', q); }
        if (cb.back[5] == q && cb.ri[1] == 'w') { right_alg(rec, cb, 'o', q); }
    }
}

static void left_alg(Record_t& rec, Cube_t& cb, char a, char c) {
    rotate_clock(rec, cb, a);
    rotate_clock(rec, cb, 'y');
    rotate_anticlock(rec, cb, a);
    white_bottom(rec, cb, c);
}

static void white_left(Record_t& rec, Cube_t& cb, char q) {
    if (cb.ri[5] == 'w' || cb.front[5] == 'w' || cb.lft[5] == 'w' || cb.back[5] == 'w') {
        if (cb.ri[5] == 'w' && cb.front[1] == q) { left_alg(rec, cb, 'r', q); }
        if (cb.front[5] == 'w' && cb.lft[1] == q) { left_alg(rec, cb, 'g', q); }
        if (cb.lft[5] == 'w' && cb.back[1] == q) { left_alg(rec, cb, 'o', q); }
        if (cb.back[5] == 'w' && cb.ri[1] == q) { left_alg(rec, cb, 'b', q); }
    }
}

static void top_alg(Record_t& rec, Cube_t& cb, char a, char b, char c) {
    rotate_anticlock(rec, cb, a);
    rotate_clock(rec, cb, 'w');
    rotate_clock(rec, cb, b);
    rotate_anticlock(rec, cb, 'w');
    white_bottom(rec, cb, c);
}

static void white_top(Record_t& rec, Cube_t& cb, char q) {
    if (cb.ri[7] == 'w' && cb.up[0] == q) { top_alg(rec, cb, 'b', 'r', q); }
    if (cb.front[7] == 'w' && cb.up[2] == q) { top_alg(rec, cb, 'r', 'g', q); }
    if (cb.lft[7] == 'w' && cb.up[4] == q) { top_alg(rec, cb, 'g', 'o', q); }
    if (cb.back[7] == 'w' && cb.up[6] == q) { top_alg(rec, cb, 'o', 'b', q); }
}

static void inv_alg(Record_t& rec, Cube_t& cb, char a, char b, char c) {
    rotate_clock(rec, cb, a);
    rotate_clock(rec, cb, b);
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, b);
    rotate_anticlock(rec, cb, a);
    white_bottom(rec, cb, c);
}

static void white_bottom_inverted(Record_t& rec, Cube_t& cb, char q) {
    if (cb.ri[3] == 'w' || cb.front[3] == 'w' || cb.lft[3] == 'w' || cb.back[3] == 'w') {
        if (cb.ri[3] == 'w' && cb.down[0] == q) { inv_alg(rec, cb, 'b', 'r', q); }
        if (cb.front[3] == 'w' && cb.down[2] == q) { inv_alg(rec, cb, 'r', 'g', q); }
        if (cb.lft[3] == 'w' && cb.down[4] == q) { inv_alg(rec, cb, 'g', 'o', q); }
        if (cb.back[3] == 'w' && cb.down[6] == q) { inv_alg(rec, cb, 'o', 'b', q); }
    }
}

static void solve_white_cross(Record_t& rec, Cube_t& cb) {
    char prefer[4] = {'b', 'r', 'g', 'o'};
    for (int i = 0; i < 4; i++) {
        if (cb.up[0] == 'w' && cb.ri[7] == prefer[i]) { rotate_clock(rec, cb, 'b'); }
        if (cb.up[2] == 'w' && cb.front[7] == prefer[i]) { rotate_clock(rec, cb, 'r'); }
        if (cb.up[4] == 'w' && cb.lft[7] == prefer[i]) { rotate_clock(rec, cb, 'g'); }
        if (cb.up[6] == 'w' && cb.back[7] == prefer[i]) { rotate_clock(rec, cb, 'o'); }
        white_bottom(rec, cb, prefer[i]);
        white_bottom_inverted(rec, cb, prefer[i]);
        white_left(rec, cb, prefer[i]);
        white_right(rec, cb, prefer[i]);
        white_top(rec, cb, prefer[i]);
        if (i != 0) { while (cb.ri[7] != 'b') { rotate_clock(rec, cb, 'w'); }}
        if (cb.up[0] == 'w' && cb.up[2] == 'w' && cb.up[4] == 'w' && cb.up[6] == 'w' && cb.ri[7] == 'b' &&
            cb.front[7] == 'r' && cb.lft[7] == 'g' && cb.back[7] == 'o') { break; }
    }
}

void white_corners_alg_left(Record_t& rec, Cube_t& cb) {
    rotate_anticlock(rec, cb, 'b');
    rotate_anticlock(rec, cb, 'y');
    rotate_clock(rec, cb, 'b');
}

void white_corners_alg_right(Record_t& rec, Cube_t& cb) {
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'y');
    rotate_anticlock(rec, cb, 'r');
}

void solve_white_corners(Record_t& rec, Cube_t& cb) {
    while (cb.front[0] != 'r' || cb.front[6] != 'r' || cb.ri[0] != 'b' || cb.ri[6] != 'b' || cb.back[0] != 'o' || cb.back[6] != 'o' ||
           cb.lft[0] != 'g' || cb.lft[6] != 'g') {
        while (cb.front[7] != 'r') {
            rotate_clock(rec, cb, 'w');
        }
        if (cb.ri[4] == 'w' || cb.front[4] == 'w' || cb.lft[4] == 'w' || cb.back[4] == 'w') {
            while (cb.ri[4] != 'w') {
                rotate_clock(rec, cb, 'y');
            }
            while (cb.front[2] != cb.front[7]) {
                rotate_clock(rec, cb, 'w');
            }
            white_corners_alg_left(rec, cb);
            while (cb.front[7] != 'r') {
                rotate_clock(rec, cb, 'w');
            }
        } else if (cb.ri[2] == 'w' || cb.front[2] == 'w' || cb.lft[2] == 'w' || cb.back[2] == 'w') {
            while (cb.front[2] != 'w') {
                rotate_clock(rec, cb, 'y');
            }
            while (cb.front[7] != cb.down[1]) {
                rotate_clock(rec, cb, 'w');
            }
            white_corners_alg_right(rec, cb);
            while (cb.front[7] != 'r') {
                rotate_clock(rec, cb, 'w');
            }
        } else if (cb.down[1] == 'w' || cb.down[3] == 'w' || cb.down[5] == 'w' || cb.down[7] == 'w') {
            while (cb.down[1] != 'w') {
                rotate_clock(rec, cb, 'y');
            }
            while (cb.front[2] != cb.ri[7]) {
                rotate_clock(rec, cb, 'w');
            }
            rotate_anticlock(rec, cb, 'b');
            rotate_clock(rec, cb, 'y');
            rotate_clock(rec, cb, 'y');
            rotate_clock(rec, cb, 'b');
            while (cb.ri[4] != 'w') {
                rotate_clock(rec, cb, 'y');
            }
            while (cb.front[2] != cb.front[7]) {
                rotate_clock(rec, cb, 'w');
            }
            white_corners_alg_left(rec, cb);
            while (cb.front[7] != 'r') {
                rotate_clock(rec, cb, 'w');
            }
        } else {
            while (cb.front[7] == cb.front[0]) {
                rotate_clock(rec, cb, 'w');
            }
            white_corners_alg_left(rec, cb);
            while (cb.front[7] != 'r') {
                rotate_clock(rec, cb, 'w');
            }
        }
    }
}

void middle_place_left_alg(Record_t& rec, Cube_t& cb, char left, char center) {
    rotate_anticlock(rec, cb, 'y');
    rotate_clock(rec, cb, left);
    rotate_clock(rec, cb, left);
    rotate_clock(rec, cb, left);
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, left);
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, center);
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, center);
}

void middle_place_right_alg(Record_t& rec, Cube_t& cb, char center, char right) {
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, right);
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, right);
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, center);
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, center);
}

void solve_middle_layer(Record_t& rec, Cube_t& cb) {
    while (cb.front[5] != 'r' || cb.front[1] != 'r' || cb.ri[1] != 'b' || cb.ri[5] != 'b' || cb.back[1] != 'o' || cb.back[5] != 'o' ||
           cb.lft[1] != 'g' || cb.lft[5] != 'g') {

        if ((cb.back[1] != 'o' && cb.lft[5] != 'g') && (cb.back[1] != 'y' || cb.lft[5] != 'y')) {
            while (cb.lft[3] != 'y' && cb.down[4] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            middle_place_right_alg(rec, cb, 'g', 'o');
        } else if ((cb.back[5] != 'o' && cb.ri[1] != 'b') && (cb.back[5] != 'y' || cb.ri[1] != 'y')) {
            while (cb.back[3] != 'y' && cb.down[6] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            middle_place_right_alg(rec, cb, 'o', 'b');
        } else if ((cb.ri[5] != 'b' && cb.front[1] != 'r') && (cb.ri[5] != 'y' || cb.front[1] != 'y')) {
            while (cb.ri[3] != 'y' && cb.down[0] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            middle_place_right_alg(rec, cb, 'b', 'r');
        } else if ((cb.front[5] != 'r' && cb.lft[1] != 'g') && (cb.front[5] != 'y' || cb.lft[1] != 'y')) {
            while (cb.front[3] != 'y' && cb.down[2] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            middle_place_right_alg(rec, cb, 'r', 'g');
        }

        while (cb.front[3] == 'y' || cb.down[2] == 'y') {
            rotate_clock(rec, cb, 'y');
        }

        if (cb.front[3] == 'r' && cb.down[2] != 'y') {
            if (cb.down[2] == 'g') {
                middle_place_right_alg(rec, cb, 'r', 'g');
            } else if (cb.down[2] == 'b') {
                middle_place_left_alg(rec, cb, 'b', 'r');
            }
        } else if (cb.front[3] == 'b' && cb.down[2] != 'y') {
            rotate_clock(rec, cb, 'y');
            if (cb.down[0] == 'r') {
                middle_place_right_alg(rec, cb, 'b', 'r');
            } else if (cb.down[0] == 'o') {
                middle_place_left_alg(rec, cb, 'o', 'b');
            }
        } else if (cb.front[3] == 'o' && cb.down[2] != 'y') {
            rotate_clock(rec, cb, 'y');
            rotate_clock(rec, cb, 'y');
            if (cb.down[6] == 'b') {
                middle_place_right_alg(rec, cb, 'o', 'b');
            } else if (cb.down[6] == 'g') {
                middle_place_left_alg(rec, cb, 'g', 'o');
            }
        } else if (cb.front[3] == 'g' && cb.down[2] != 'y') {
            rotate_anticlock(rec, cb, 'y');
            if (cb.down[4] == 'o') {
                middle_place_right_alg(rec, cb, 'g', 'o');
            } else if (cb.down[4] == 'r') {
                middle_place_left_alg(rec, cb, 'r', 'g');
            }
        }
    }
}

void yellow_cross_algorithm(Record_t& rec, Cube_t& cb) {
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, 'g');
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, 'g');
    rotate_anticlock(rec, cb, 'r');
}

void solve_yellow_cross(Record_t& rec, Cube_t& cb) {
    while (cb.down[0] != 'y' || cb.down[2] != 'y' || cb.down[4] != 'y' || cb.down[6] != 'y') {
        if ((cb.down[0] == 'y' && cb.down[6] == 'y') || (cb.down[4] == 'y' && cb.down[6] == 'y')
            || (cb.down[2] == 'y' && cb.down[4] == 'y') || (cb.down[0] == 'y' && cb.down[2] == 'y')) {
            while (cb.down[0] != 'y' && cb.down[6] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_cross_algorithm(rec, cb);
        }
        if ((cb.down[2] == 'y' && cb.down[6] == 'y') || (cb.down[0] == 'y' && cb.down[4] == 'y')) {
            while (cb.down[0] != 'y' && cb.down[4] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_cross_algorithm(rec, cb);
            yellow_cross_algorithm(rec, cb);
        } else if (cb.down[8] == 'y') {
            yellow_cross_algorithm(rec, cb);
            rotate_clock(rec, cb, 'y');
            yellow_cross_algorithm(rec, cb);
            yellow_cross_algorithm(rec, cb);
        }
    }
}

void yellow_corners_algorithm(Record_t& rec, Cube_t& cb) {
    rotate_clock(rec, cb, 'g');
    rotate_clock(rec, cb, 'y');
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, 'g');
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, 'y');
    rotate_anticlock(rec, cb, 'g');
}

void solve_yellow_corners(Record_t& rec, Cube_t& cb) {
    while (cb.down[1] != 'y' || cb.down[3] != 'y' || cb.down[5] != 'y' || cb.down[7] != 'y') {
        if ((cb.down[1] == 'y' && cb.down[3] != 'y' && cb.down[5] != 'y' && cb.down[7] != 'y')
            || (cb.down[3] == 'y' && cb.down[1] != 'y' && cb.down[5] != 'y' && cb.down[7] != 'y')
            || (cb.down[5] == 'y' && cb.down[1] != 'y' && cb.down[3] != 'y' && cb.down[7] != 'y')
            || (cb.down[7] == 'y' && cb.down[1] != 'y' && cb.down[3] != 'y' && cb.down[5] != 'y')) {
            while (cb.down[1] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        } else if ((cb.lft[2] == 'y' && cb.lft[4] == 'y' && cb.down[1] == 'y' && cb.down[7] == 'y')
                   || (cb.back[2] == 'y' && cb.back[4] == 'y' && cb.down[1] == 'y' && cb.down[3] == 'y')
                   || (cb.ri[2] == 'y' && cb.ri[4] == 'y' && cb.down[3] == 'y' && cb.down[5] == 'y')
                   || (cb.front[2] == 'y' && cb.front[4] == 'y' && cb.down[5] == 'y' && cb.down[7] == 'y')) {
            while (cb.front[2] != 'y' && cb.front[4] != 'y' && cb.down[5] != 'y' && cb.down[7] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        } else if ((cb.front[4] == 'y' && cb.back[2] == 'y' && cb.down[1] == 'y' && cb.down[7] == 'y')
                   || (cb.ri[2] == 'y' && cb.lft[4] == 'y' && cb.down[1] == 'y' && cb.down[3] == 'y')
                   || (cb.front[2] == 'y' && cb.back[4] == 'y' && cb.down[3] == 'y' && cb.down[5] == 'y')
                   || (cb.ri[4] == 'y' && cb.lft[2] == 'y' && cb.down[5] == 'y' && cb.down[7] == 'y')) {
            while (cb.front[2] != 'y' && cb.back[4] != 'y' && cb.down[3] != 'y' && cb.down[5] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        } else if ((cb.lft[2] == 'y' && cb.lft[4] == 'y' && cb.ri[2] == 'y' && cb.ri[4] == 'y')
                   || (cb.front[2] == 'y' && cb.front[4] == 'y' && cb.back[2] == 'y' && cb.back[4] == 'y')) {
            while (cb.lft[2] != 'y' && cb.lft[4] != 'y' && cb.ri[2] != 'y' && cb.ri[4] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        } else if ((cb.lft[2] == 'y' && cb.back[2] == 'y' && cb.back[4] == 'y' && cb.ri[4] == 'y')
                   || (cb.front[4] == 'y' && cb.back[2] == 'y' && cb.ri[2] == 'y' && cb.ri[4] == 'y')
                   || (cb.front[2] == 'y' && cb.front[4] == 'y' && cb.lft[4] == 'y' && cb.ri[2] == 'y')
                   || (cb.lft[2] == 'y' && cb.lft[4] == 'y' && cb.back[4] == 'y' && cb.front[2] == 'y')) {
            while (cb.lft[2] != 'y' && cb.back[2] != 'y' && cb.back[4] != 'y' && cb.ri[4] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        } else if ((cb.down[1] == 'y' && cb.down[5] == 'y' && cb.down[3] != 'y' && cb.down[7] != 'y')
                   || (cb.down[3] == 'y' && cb.down[7] == 'y' && cb.down[1] != 'y' && cb.down[5] != 'y')) {
            while (cb.front[2] != 'y' && cb.lft[4] != 'y') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corners_algorithm(rec, cb);
        }
    }
}

void yellow_corner_orientation_algorithm(Record_t& rec, Cube_t& cb) {
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'r');
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'o');
    rotate_clock(rec, cb, 'o');
    rotate_clock(rec, cb, 'g');
    rotate_anticlock(rec, cb, 'r');
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'o');
    rotate_clock(rec, cb, 'o');
    rotate_clock(rec, cb, 'g');
    rotate_clock(rec, cb, 'g');
    rotate_anticlock(rec, cb, 'y');
}

void yellow_corner_orientation(Record_t& rec, Cube_t& cb) {
    while (cb.front[2] != 'r' || cb.front[4] != 'r' || cb.lft[2] != 'g' || cb.lft[4] != 'g'
           || cb.back[2] != 'o' || cb.back[4] != 'o' || cb.ri[2] != 'b' || cb.ri[4] != 'b') {
        if ((cb.front[2] == cb.front[4]) || (cb.lft[2] == cb.lft[4]) || (cb.back[2] == cb.back[4]) || (cb.ri[2] == cb.ri[4])) {
            while (cb.back[2] != cb.back[4]) {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corner_orientation_algorithm(rec, cb);
            while (cb.ri[2] != 'b') {
                rotate_clock(rec, cb, 'y');
            }
        } else {
            while (cb.back[4] != 'o' && cb.front[4] != 'r') {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corner_orientation_algorithm(rec, cb);
            while (cb.back[2] != cb.back[4]) {
                rotate_clock(rec, cb, 'y');
            }
            yellow_corner_orientation_algorithm(rec, cb);
            while (cb.ri[2] != 'b') {
                rotate_clock(rec, cb, 'y');
            }
        }
    }
}

void yellow_edges_colour_arrangement_right(Record_t& rec, Cube_t& cb) {
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
    rotate_anticlock(rec, cb, 'y');
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'b');
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
    rotate_anticlock(rec, cb, 'b');
    rotate_clock(rec, cb, 'g');
    rotate_anticlock(rec, cb, 'y');
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
}

void yellow_edges_colour_arrangement_left(Record_t& rec, Cube_t& cb) {
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, 'b');
    rotate_anticlock(rec, cb, 'g');
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
    rotate_anticlock(rec, cb, 'b');
    rotate_clock(rec, cb, 'g');
    rotate_clock(rec, cb, 'y');
    rotate_clock(rec, cb, 'r');
    rotate_clock(rec, cb, 'r');
}

void yellow_edges_colour_arrangement(Record_t& rec, Cube_t& cb) {
    while (cb.front[2] != 'r') {
        rotate_clock(rec, cb, 'r');
    }
    if (cb.front[3] == 'o' && cb.back[3] == 'r' && cb.ri[3] == 'g' && cb.lft[3] == 'b') {
        yellow_edges_colour_arrangement_left(rec, cb);
    } else if (cb.front[3] == 'b' && cb.ri[3] == 'r') {
        yellow_edges_colour_arrangement_left(rec, cb);
    } else if (cb.front[3] == 'g' && cb.lft[3] == 'r') {
        yellow_edges_colour_arrangement_left(rec, cb);
    }
    while (cb.back[2] != cb.back[3]) {
        rotate_clock(rec, cb, 'y');
    }
    if (cb.front[3] == cb.lft[2]) {
        yellow_edges_colour_arrangement_right(rec, cb);
    } else if (cb.front[3] == cb.ri[2]) {
        yellow_edges_colour_arrangement_left(rec, cb);
    }
    while (cb.front[3] != 'r') {
        rotate_clock(rec, cb, 'y');
    }
}

// UFBRLD
Cube_t create_cube(const string& str) {
    Cube_t cb;
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 9; j++) {
            cb.cb[i][j] = str[i * 9 + j];
        }
    }
    return cb;
}

string _solve_white_cross(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    solve_white_cross(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _solve_white_corners(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    solve_white_corners(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _solve_middle_layer(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    solve_middle_layer(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _solve_yellow_cross(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    solve_yellow_cross(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _solve_yellow_corners(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    solve_yellow_corners(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _yellow_corner_orientation(const string& cube_str) {
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    yellow_corner_orientation(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _yellow_edges_colour_arrangement(const string& cube_str){
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    yellow_edges_colour_arrangement(rec, cb);
    return rec.to_string() + "\t" + cb.to_string();
}

string _partial_solve(const string& cube_str, int steps) {
    static const function<void(Record_t&, Cube_t&)> f[] = {
           solve_white_cross,
           solve_white_corners,
           solve_middle_layer,
           solve_yellow_cross,
           solve_yellow_corners,
           yellow_corner_orientation,
           yellow_edges_colour_arrangement
    };
    Cube_t cb = create_cube(cube_str);
    Record_t rec;
    for(int i = 0; i < steps; i++) {
        f[i](rec, cb);
    }
    return rec.to_string() + "\t" + cb.to_string();
}

string _solve(const string& cube_str) {
    return _partial_solve(cube_str, 7);
}

#ifndef DEBUG
PYBIND11_MODULE(lblcube, m) {
    m.def("solve_white_cross", &_solve_white_cross, "");
    m.def("solve_white_corners", &_solve_white_corners, "");
    m.def("solve_middle_layer", &_solve_middle_layer, "");
    m.def("solve_yellow_cross", &_solve_yellow_cross, "");
    m.def("solve_yellow_corners", &_solve_yellow_corners, "");
    m.def("yellow_corner_orientation", &_yellow_corner_orientation, "");
    m.def("yellow_edges_colour_arrangement", &_yellow_edges_colour_arrangement, "");
    m.def("solve", &_solve, "");
    m.def("partial_solve", &_partial_solve, "");
}
#endif

int main() {
    string ans = _solve("wwwooowwwrrrrwwwrryyyoooooobbbbbbbbbgggggggggyyyrrryyy");
    cout << ans << endl;
    return 0;
}