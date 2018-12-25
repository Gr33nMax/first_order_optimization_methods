#!/usr/bin/env python
# Author: Sidorenko Maxim
# Автор: Сидоренко Максим
# vk.com/maksim2009rus
# -----------------------------------------------------------
# Search for a priori segment of uncertainty,
# implementation of methods of division of the segment in half,
# the Golden ratio
# to find the minimum given as a polynomial function
# -----------------------------------------------------------
# (Поиск априрного отрезка неопределенности,
#  реализация методов деления отрезка пополам, золотого сечения
#  для поиска минимума у заданной в виде полинома функции)
# -----------------------------------------------------------

import numpy as np
import re
import pylab
from matplotlib import mlab

# Setting the accuracy of calculations
# (Задание точности вычислений)
EPS = 0.0005

# Initialization of the polynomial
# (Задание многочлена)
POLYNOM = "0.1*x^2 - 20*x + 970"
# POLYNOM = "x^2 - 10x + 21"
# POLYNOM = "-x^3 - x^2 + 5x + 1"
POLYNOM = POLYNOM.replace(' ', '')


class PolynomMethods:
    @staticmethod
    def f(t: float) -> float:
        """
        Calculating the value of a polynomial 'p' at point t
        Вычисление значения многочлена 'p' в точке t
        """
        p = PolynomMethods.coefficients_input(POLYNOM)
        return np.polyval(p, t)

    @staticmethod
    def segment_calculation(n: int = 32) -> tuple:
        """
        The calculation a priori of the segment
        (Вычисление априорного отрезка)
        """
        # Determine the direction of the search
        # (Определение направления поиска)
        f = PolynomMethods.f
        direction = ""  # left or right (влево или вправо)
        x0 = 0  # start point (начальная точка отсчета)
        delta = 1  # increment (приращение)

        # the value at the point x0 (значение в точке x0)
        f_x0 = f(x0)
        # the value at the point (x0-delta) (значение в точке (x0-delta))
        x1 = x0 - delta
        f_x1 = f(x1)
        # the value at the point (x0+delta) (значение в точке (x0+delta))
        x2 = x0 + delta
        f_x2 = f(x2)
        if f_x1 > f_x0 > f_x2:
            direction = "right"  # движение по функции вправо
        elif f_x1 < f_x0 < f_x2:
            direction = "left"  # движение по функции влево
            delta = -delta
        elif f_x1 > f_x0 < f_x2:
            return [x1, x2]
        elif f_x1 < f_x0 > f_x2:
            return ()

        # Segment search (Поиск отрезка)
        multiplier = 2  # множитель
        x_prev = x0  # the previous value of x (предыдущее значение x)
        x_prev_prev = x0
        for i in range(n):
            if i > n:
                print('The number of iterations exceeded '
                      '(Превышено число итераций).')
                return ()

            x = x_prev + multiplier * delta
            if f(x) > f(x_prev) and direction == "right":
                return [x_prev_prev, x], i
            elif f(x) < f(x_prev) and direction == "left":
                return [x, x_prev_prev], i
            elif f(x) == f(x_prev):
                ans = (x + x_prev) / 2 - delta, (x + x_prev) / 2 + delta
                if ans[0] > ans[1]:
                    ans[0], ans[1] = ans[1], ans[0]
                return [ans[0], ans[1]], i

            x_prev_prev = x_prev
            x_prev = x
            multiplier *= 2
            i += 1

    @staticmethod
    def plot_drawing(p: np.poly1d, segment: list) -> None:
        """
        The function displays a segment as a graph
        (Функция выводит на экран найденный отрезок в виде графика)
        """
        xmin = segment[0]
        xmax = segment[1]
        dx = 0.01
        xlist = mlab.frange(xmin, xmax, dx)
        ylist = [PolynomMethods.f(x) for x in xlist]
        pylab.plot(xlist, ylist)
        pylab.show(block=False)

    @staticmethod
    def search_min_2points(segment: list, eps: float = 0.001,
                           n: int = 1000) -> tuple:
        """
        2-point search for the minimum by dividing the segment in half
        (2-х точечный поиск минимума методом деления отрезка пополам)
        """
        f = PolynomMethods.f
        a, b = segment[0], segment[1]
        delta = 0.0001
        i = 0
        while abs(b - a) >= eps and i < n:
            i += 1
            x1 = (a + b - delta) / 2
            f1 = f(x1)
            x2 = (a + b + delta) / 2
            f2 = f(x2)
            if f1 < f2:
                b = x2
            elif f1 > f2:
                a = x1
            else:
                break
        return (a + b) / 2, i

    @staticmethod
    def search_min_3points(segment: list,
                           eps: float = 0.001) -> tuple:
        """
        3-point search for the minimum by dividing the segment in half
        (3-х точечный поиск минимума методом деления отрезка пополам)
        """
        f = PolynomMethods.f

        a, b = segment[0], segment[1]
        L = b - a
        i = 0
        while abs(L) >= eps:
            i += 1
            L = b - a
            x2 = (a + b) / 2
            f2 = f(x2)
            x1 = a + L / 4
            f1 = f(x1)
            x3 = b - L / 4
            f3 = f(x3)

            if f1 < f2:
                b = x2
                x2 = x1
            else:
                if f2 > f3:
                    a = x2
                    x2 = x3
                else:
                    a = x1
                    b = x3
            L = b - a
        return (a + b) / 2, i

    @staticmethod
    def search_min_golden_ratio(segment: list, eps: float = 0.001) -> tuple:
        """
        Implementation of the method of the Golden ratio
        (Реализация метода золотого сечения)
        """
        f = PolynomMethods.f

        a, b = segment[0], segment[1]
        phi = (1 + np.sqrt(5)) / 2  # golden ratio (золотое сечение)
        i = 0
        while abs(b - a) >= eps:
            i += 1
            x1 = b - (b - a) / phi
            x2 = a + (b - a) / phi
            y1, y2 = f(x1), f(x2)
            if y1 >= y2:
                a = x1
            else:
                b = x2
        return (a + b) / 2, i

    @staticmethod
    def search_min_fibonacci(segment: list, n: int = 30) -> tuple:
        """
        Implementation of the Fibonacci method for finding the minimum
        (Реализация метода Фибоначчи для поиска минимума)
        """
        f = PolynomMethods.f
        def fib(q: int) -> int:
            """
            Effective Fibonacci function
            (Эффективная генерация чисел Фибоначчи)
            """
            return pow(2 << q, q + 1, (4 << 2 * q) - (2 << q) - 1) % (2 << q)

        i = n  # number of iterations (количество итераций)
        a, b = segment[0], segment[1]
        x1 = a + (b - a) * (fib(n - 2) / fib(n))
        x2 = a + (b - a) * (fib(n - 1) / fib(n))
        y1, y2 = f(x1), f(x2)
        while n > 1:
            if y1 > y2:
                a = x1
                x1 = x2
                x2 = b - (x1 - a)
                y1 = y2
                y2 = f(x2)
            else:
                b = x2
                x2 = x1
                x1 = a + (b - x2)
                y2 = y1
                y1 = f(x1)
            n = n - 1

        return (x1 + x2) / 2, i

    @staticmethod
    def coefficients_input(p: np.poly1d) -> list:
        """
        Identification of the polynomial coefficients of the POLYNOM string
        using regular expression
        (Вычленение коэффициентов многочлена из строки POLYNOM при помощи
        регулярного выражения)
        """
        found = re.findall(
            r'([-+]?\d*\.?\d*)?\*?x(\^\d+)?|([-+]?\d+\.?\d*)(?!x)',
            p
        )
        match = [list(s) for s in found]
        for s in match:
            temp = re.match(r'[-+]?\d+\.?\d*', s[0])
            if not s[1]:  # 'x' or constant
                if not temp and not s[2]:  # 'x'
                    s[0], s[1], s[2] = 1, '^1', 0
                elif not temp and s[2]:  # constant
                    s[0], s[1] = 1, '^0'
                elif temp and not s[2]:
                    s[0], s[1], s[2] = float(temp.group()), '^1', 0
            else:  # 'x^degree'
                s[2] = 0
                if not temp:
                    if s[0] == '-':
                        s[0] = -1
                    else:
                        s[0] = 1
                else:
                    s[0] = float(temp.group())
        match.sort(key=lambda f: f[1], reverse=True)
        coefficients = list()
        for s in match:
            if not s[2]:
                coefficients.append(s[0])
            else:
                coefficients.append(float(s[2]))
        return tuple(coefficients)

    @staticmethod
    def search_min_quadratic(segment: list, eps: int=0.001) -> tuple:
        """
        Implementation of the Quadratic interpolation method for
        finding the minimum
        (Реализация метода квадратичной интерполяции для поиска минимума)
        """
        f = PolynomMethods.f
        h = 0.0005
        a, b = segment[0], segment[1]
        i = 0
        x0 = a
        xmin = x0 + 5
        while abs(f(x0) - f(xmin)) >= eps:
            i += 1
            x0 = (6*a + b) / 4
            x1 = x0 - h
            x2 = x0
            x3 = x0 + h
            y1, y2, y3 = f(x1), f(x2), f(x3)
            xmin = 0.5 * ((x2 * x2 - x3 * x3) * f(x1) + (x3 * x3 - x1 * x1) *
                          f(x2) + (x1 * x1 - x2 * x2) * f(x3)) / \
                   ((x2 - x3) * f(x1) + (x3 - x1) * f(x2) +
                    (x1 - x2) * f(x3))
            x0 = xmin
        return x0, i


def main():
    # The calculation of the boundaries of priori interval of uncertainty
    # (Определение границ априорного интервала неопределенности)
    p = PolynomMethods()
    p.coefficients = p.coefficients_input(POLYNOM)
    p.polynom = np.poly1d(p.coefficients)
    segment = p.segment_calculation(32)
    if not segment:
        print('The function is non-unimodal (Функция неунимодальна).')
        exit(0)
    print('A priori segment of uncertainty '
          '(Априорный отрезок неопределенности): {}.'.format(segment[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(segment[1]))
    p.plot_drawing(p.polynom, segment[0])
    # Finding the minimum (нахождение минимума)
    # 2-point method of dividing the segment in half
    # (2-х точечный метод деления отрезка пополам)
    min_2points = p.search_min_2points(segment[0], EPS)
    print('The minimum on the 2-point method '
          '(Минимум по 2-х точечному методу): {}.'.format(min_2points[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(min_2points[1]))
    # 3-point method of dividing the segment in half
    # (3-х точечный метод деления отрезка пополам)
    min_3points = p.search_min_3points(segment[0], EPS)
    print('The minimum on the 3-point method '
          '(Минимум по 3-х точечному методу): {}.'.format(min_3points[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(min_3points[1]))
    # The golden ratio method (метод золотого сечениия)
    min_g_ratio = p.search_min_golden_ratio(segment[0], EPS)
    print('Minimum by the method of Golden ratio '
          '(Минимум по методу золотого сечения): {}.'.format(min_g_ratio[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(min_g_ratio[1]))
    # Fibonacci method (метод Фибоначчи)
    min_fib = p.search_min_fibonacci(segment[0])
    print('Minimum by the method of Fibonacci '
          '(Минимум по методу Фибоначчи): {}.'.format(min_fib[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(min_fib[1]))
    # Quadratic interpolation method (метод квадратичной интерполяции)
    min_quadratic = p.search_min_quadratic(segment[0])
    print('Minimum by the method of quadratic interpolation '
          '(Минимум по методу квадратичной интерполяции): {}.'
          .format(min_quadratic[0]))
    print('Number of iterations '
          '(Количество итераций): {}.\n'.format(min_quadratic[1]))
    pylab.show()

if __name__ == '__main__':
    main()
