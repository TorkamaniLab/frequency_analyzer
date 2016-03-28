""" A collection of matrix utilities.

author: Brian Schrader
since: 2015-07-21
"""


from math import cos, sin


R = {
        'x': [['1', '0', '0'],
              ['0', 'cos(ang)', '-sin(ang)'],
              ['0', 'sin(ang)', 'cos(ang)']
            ],
        'y': [['cos(ang)', '0', 'sin(ang)'],
              ['0', '1', '0'],
              ['-sin(ang)', '0', 'cos(ang)']
            ],
        'z': [['cos(ang)', '-sin(ang)', '0'],
              ['sin(ang)', 'cos(ang)', '0'],
              ['0', '0', '1']
            ]
    }


def transformation_matrix(angles):
    """ Given an set of transformation angles,
    generate a series of transformation matrices
    """
    R_t = []
    for axis, ang in angles:
        r_t = []
        for row in R[axis]:
            r_t_i = []
            for el in row:
                r_t_i.append(eval(el))
            r_t.append(r_t_i)
        R_t.append(r_t)
    return R_t


def transform(v, R):
    """ Given a 3d vector and a list of transformation
    matrices, transpose the given vector using those matrices.
    """
    V = [0, 0, 0]
    x, y, z = v
    for r_x in R:
        V_i = []
        for a, b, c in r_x:
            p_i = a*x + b*y + c*z
            V_i.append(p_i)
        V[0] += V_i[0]
        V[1] += V_i[1]
        V[2] += V_i[2]
    return V

