import PySimpleGUI as sg
import numpy as np
import math as m
from scipy.optimize import fsolve


# ======================================================================================================
#                                   Define equations
# ======================================================================================================

def equations_cubic(lp):
    a = lp[0]  # special case: only one variable
    return [
        1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - (h[i] ** 2 + k[i] ** 2 + l[i] ** 2) / a ** 2
        for i in range(len(two_theta))]


def equations_hexagonal(lp):
    a, c = lp
    return [1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - 4 / 3 * (
                h[i] ** 2 + h[i] * k[i] + k[i] ** 2) / a ** 2 - l[i] ** 2 / c ** 2
            for i in range(len(two_theta))]


def equations_rhombohedral(lp):
    a, alpha = lp
    return [1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - (
                (h[i] ** 2 + k[i] ** 2 + l[i] ** 2) * (m.sin(m.radians(alpha))) ** 2 + 2 * (
                    h[i] * k[i] + k[i] * l[i] + h[i] * l[i]) * (
                            (m.cos(m.radians(alpha))) ** 2 - (m.cos(m.radians(alpha))))) / a ** 2 / (
                        1 - 3 * (m.cos(m.radians(alpha))) ** 2 + 2 * (m.cos(m.radians(alpha))) ** 3)
            for i in range(len(two_theta))]


def equations_tetragonal(lp):
    a, c = lp
    return [1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - (h[i] ** 2 + k[i] ** 2) / a ** 2 - l[
        i] ** 2 / c ** 2
            for i in range(len(two_theta))]


def equations_orthorhombic(lp):
    a, b, c = lp
    return [
        1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - h[i] ** 2 / a ** 2 - k[i] ** 2 / b ** 2 - l[
            i] ** 2 / c ** 2
        for i in range(len(two_theta))]


def equations_monoclinic(lp):
    a, b, c, beta = lp
    return [1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2
            - h[i] ** 2 / (a * m.sin(m.radians(beta))) ** 2
            - k[i] ** 2 / b ** 2
            - l[i] ** 2 / (c * m.sin(m.radians(beta))) ** 2
            + (2 * h[i] * l[i] * (m.cos(m.radians(beta)))) / (a * c * (m.sin(m.radians(beta))) ** 2)
            for i in range(len(two_theta))]


def equations_triclinic(lp):
    a, b, c, alpha, beta, gamma = lp
    s11 = b ** 2 * c ** 2 * m.sin(m.radians(alpha)) ** 2
    s22 = a ** 2 * c ** 2 * m.sin(m.radians(beta)) ** 2
    s33 = a ** 2 * b ** 2 * m.sin(m.radians(gamma)) ** 2
    s12 = a * b * c ** 2 * (m.cos(m.radians(alpha)) * m.cos(m.radians(beta)) - m.cos(m.radians(gamma)))
    s23 = a ** 2 * b * c * (m.cos(m.radians(beta)) * m.cos(m.radians(gamma)) - m.cos(m.radians(alpha)))
    s13 = a * b ** 2 * c * (m.cos(m.radians(gamma)) * m.cos(m.radians(alpha)) - m.cos(m.radians(beta)))
    V = a * b * c * (
                1 - m.cos(m.radians(alpha)) ** 2 - m.cos(m.radians(beta)) ** 2 - m.cos(m.radians(gamma)) ** 2 + 2 * (
                    m.cos(m.radians(alpha)) * m.cos(m.radians(beta)) * m.cos(m.radians(gamma)))) ** 0.5

    return [1 / (wavelenght / (2 * m.sin(m.radians(two_theta[i] / 2)))) ** 2 - 1 / V ** 2 * (
                s11 * h[i] ** 2 + s22 * k[i] ** 2 + s33 * l[i] ** 2 + 2 * s12 * h[i] * k[i] + 2 * s23 * k[i] * l[
            i] + 2 * s13 * h[i] * l[i])
            for i in range(len(two_theta))]


# ======================================================================================================
#                                   Define solving function
# ======================================================================================================

def solve_equations(lattice_system):
    if lattice_system == 'Cubic':
        # solving cubic
        a = fsolve(equations_cubic, [a_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a[0], a[0], a[0]),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha_start, beta_start, gamma_start),
                 'Volume: {0:.2f} Å3'.format(a[0] ** 3),
                 'Residuals: a: {0:.5f} Å'.format(equations_cubic((a))[0])
                 )

    elif lattice_system == 'Hexagonal':
        # solving hexagonal
        a, c = fsolve(equations_hexagonal, [a_start, c_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, a, c),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha_start, beta_start, 120),
                 'Volume: {0:.2f} Å3'.format(a ** 2 * c * m.sin(m.radians(60))),
                 'Residuals: a: {0:.5f} Å, c: {1:.5f} Å'.format(equations_hexagonal((a, c))[0],
                                                                equations_hexagonal((a, c))[1])
                 )

    elif lattice_system == 'Rhombohedral':
        # solving rhombohedral
        a, alpha = fsolve(equations_rhombohedral, [a_start, alpha_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, a, a),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha, alpha, alpha),
                 'Volume: {0:.2f} Å3'.format(
                     a ** 3 * (1 - 3 * (m.cos(m.radians(alpha))) ** 2 + 2 * (m.cos(m.radians(alpha))) ** 3) ** 0.5),
                 'Residuals: a: {0:.5f} Å, c: {1:.5f} Å'.format(equations_rhombohedral((a, alpha))[0],
                                                                equations_rhombohedral((a, alpha))[1])
                 )

    elif lattice_system == 'Tetragonal':
        # solving tetragonal
        a, c = fsolve(equations_tetragonal, [a_start, c_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, a, c),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha_start, beta_start, gamma_start),
                 'Volume: {0:.2f} Å3'.format(a ** 2 * c),
                 'Residuals: a: {0:.5f} Å, c: {1:.5f} Å'.format(equations_tetragonal((a, c))[0],
                                                                equations_tetragonal((a, c))[1])
                 )

    elif lattice_system == 'Orthorhombic':
        # solving orthorhombic
        a, b, c = fsolve(equations_orthorhombic, [a_start, b_start, c_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, b, c),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha_start, beta_start, gamma_start),
                 'Volume: {0:.2f} Å3'.format(a * b * c),
                 'Residuals: a: {0:.5f} Å, b: {1:.5f} Å, c: {2:.5f} Å'.format(equations_orthorhombic((a, b, c))[0],
                                                                              equations_orthorhombic((a, b, c))[1],
                                                                              equations_orthorhombic((a, b, c))[2])
                 )

    elif lattice_system == 'Monoclinic':
        # solving monoclinic
        a, b, c, beta = fsolve(equations_monoclinic, [a_start, b_start, c_start, beta_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, b, c),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha_start, beta, gamma_start),
                 'Volume: {0:.2f} Å3'.format(a * b * c * m.sin(m.radians(beta))),
                 'Residuals: a: {0:.5f} Å, b: {1:.5f} Å, c: {2:.5f} Å, beta: {3:.5f}°'.format(
                     equations_monoclinic((a, b, c, beta))[0], equations_monoclinic((a, b, c, beta))[1],
                     equations_monoclinic((a, b, c, beta))[2], equations_monoclinic((a, b, c, beta))[3])
                 )

    elif lattice_system == 'Triclinic':
        # solving triclinic
        a, b, c, alpha, beta, gamma = fsolve(equations_triclinic,
                                             [a_start, b_start, c_start, alpha_start, beta_start, gamma_start])

        # print solution
        sg.popup('Solution',
                 'a: {0:.2f} Å, b: {1:.2f} Å, c: {2:.2f} Å'.format(a, b, c),
                 'alpha: {0:.2f}°, beta: {1:.2f}°, gamma: {2:.2f}°'.format(alpha, beta, gamma),
                 'Volume: {0:.2f} Å3'.format(a * b * c * (
                             1 - m.cos(m.radians(alpha)) ** 2 - m.cos(m.radians(beta)) ** 2 - m.cos(
                         m.radians(gamma)) ** 2 + 2 * (m.cos(m.radians(alpha)) * m.cos(m.radians(beta)) * m.cos(
                         m.radians(gamma)))) ** 0.5),
                 'Residuals: a: {0:.5f} Å, b: {1:.5f} Å, c: {2:.5f} Å, alpha: {3:.5f}°, beta: {4:.5f}°, gamma: {5:.5f}°'.format(
                     equations_triclinic((a, b, c, alpha, beta, gamma))[0],
                     equations_triclinic((a, b, c, alpha, beta, gamma))[1],
                     equations_triclinic((a, b, c, alpha, beta, gamma))[2],
                     equations_triclinic((a, b, c, alpha, beta, gamma))[3],
                     equations_triclinic((a, b, c, alpha, beta, gamma))[4],
                     equations_triclinic((a, b, c, alpha, beta, gamma))[5])
                 )


# initial guess for solution
a_start, b_start, c_start, alpha_start, beta_start, gamma_start = [1, 1, 1, 90, 90, 90]

# ======================================================================================================
#                                        Start GUI
# ======================================================================================================
sg.change_look_and_feel('DefaultNoMoreNagging')

layout = [
    [sg.Text('Select Lattice System:')],
    [sg.Radio('Triclinic', "RADIO1", default=True, key='Triclinic')],
    [sg.Radio('Monoclinic', "RADIO1", key='Monoclinic')],
    [sg.Radio('Orthorhombic', "RADIO1", key='Orthorhombic')],
    [sg.Radio('Tetragonal', "RADIO1", key='Tetragonal')],
    [sg.Radio('Rhombohedral', "RADIO1", key='Rhombohedral')],
    [sg.Radio('Hexagonal', "RADIO1", key='Hexagonal')],
    [sg.Radio('Cubic', "RADIO1", key='Cubic')],
    [sg.Button('Select'), sg.Cancel()]
]

number_of_peaks = {'Triclinic': 6, 'Monoclinic': 4, 'Orthorhombic': 3, 'Tetragonal': 2, 'Rhombohedral': 2,
                   'Hexagonal': 2, 'Cubic': 1}

window = sg.Window('Unit Cell Calculator', layout, default_element_size=(40, 1), grab_anywhere=False)

while True:
    event, values = window.read()
    if event == 'Select':
        window.close()
        # print(event)
        # print(values)
        lattice_system = list(values.keys())[list(values.values()).index(True)]
        # new layout
        layout = [
            [sg.Text('Lattice System:')],
            [sg.Text(lattice_system)],
            [sg.Text('Wavelength:'), sg.In(default_text='1.5406', size=(15, 1)), sg.Text('Å')],
            [sg.Text('2 Theta', size=(22, 1)), sg.Text('h', size=(10, 1)), sg.Text('k', size=(10, 1)),
             sg.Text('l', size=(10, 1))],
        ]
        for i in range(number_of_peaks[lattice_system]):
            layout.append([sg.In(size=(10, 1)), sg.Text('°', size=(5, 1)), sg.In(size=(10, 1)), sg.In(size=(10, 1)),
                           sg.In(size=(10, 1))])
        layout.append([sg.Button('Calculate'), sg.Cancel()])

        # create new window
        window = sg.Window('Unit Cell Calculator', layout, default_element_size=(40, 1), grab_anywhere=False)
        while True:
            event, values = window.read()
            if event == 'Calculate':
                # print(event)
                # print(values)
                try:
                    wavelenght = float(values[0])
                except ValueError:
                    sg.popup('Error', 'wavelenght value must be float')
                    continue
                input_values = np.reshape(list(values.values())[1:], (number_of_peaks[lattice_system], 4))
                try:
                    two_theta = input_values[:, 0].astype(float)
                except ValueError:
                    sg.popup('Error', 'two theta values must be float')
                    continue
                try:
                    h = input_values[:, 1].astype(int)
                except ValueError:
                    sg.popup('Error', 'h values must be integer')
                    continue
                try:
                    k = input_values[:, 2].astype(int)
                except ValueError:
                    sg.popup('Error', 'k values must be integer')
                    continue
                try:
                    l = input_values[:, 3].astype(int)
                except ValueError:
                    sg.popup('Error', 'l values must be integer')
                    continue

                # solve
                solve_equations(lattice_system)
            else:
                window.close()
                break
    else:
        window.close()
        break