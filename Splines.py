import numpy as np
from scipy.linalg import solve_banded

class CSpline:
    def __init__(self, xi, yi, si):
        """Interpolación cúbica spline asumiendo que `xi` es una secuencia
        ordenada, `yi` son los puntos a interpolar y `si` son los
        valores de la derivada de `yi`. En cada segmento definido por
        `xi`, se usa un polinomio de la forma:

        pi(x) = yi + si (x-xi) + ai(x-xi)**2 + bi(x-xi)**3

        de modo que `pi(xi)=yi` y `pi'(xi)=si`. Los coeficientes `ai`
        y `bi` se encuentran imponiendo que `pi(x[i+1])=y[i+1]` y
        `pi'(x[i+1])=s[i+1]`.

        Argumentos
        ----------
        xi : (iterable) Lista ordenada de puntos, correspondientes a los nodos.
        yi : (iterable) Puntos a interpolar.
        si : (iterable) Derivada en cada nodo.

        Ejemplo
        -------
        p = CSpline(xi, yi, si)  # calcula coeficientes usando xi, yi, si
        p(x)                     # evalua en un cierto valor de x
        """

        # Guarda los nodos y la derivada
        self.xi = np.asarray(xi)
        self.yi = np.asarray(yi)
        self.si = np.asarray(si)

        # calcula diferencia entre nodos
        dx = np.diff(self.xi)
        dy = np.diff(self.yi)

        self.ai = (3*dy/dx - self.si[1:] - 2*self.si[:-1]) / dx
        self.bi = (self.si[1:] + self.si[:-1] - 2*dy/dx) / dx**2


    def __call__(self, x):
        """Evalúa la cspline interpolante en un valor de `x`."""

        # busca el segmento `i` tal que `xi[i] <= x < xi[i+1]`
        i = np.searchsorted(self.xi, x, side="right") - 1

        # si hay elementos de `x<min(xi)`e o `x>max(xi)`, retorna `0`
        # o `N-2`, respectivamente, con `N=len(xi)`
        i = np.clip(i, 0, self.xi.size-2)

        dx = x - self.xi[i]

        return self.yi[i] + dx * (self.si[i] + dx * (self.ai[i] + dx * self.bi[i]))


class CSpline_Clasica(CSpline):
    def __init__(self, xi, yi):
        """Interpolación cúbica spline asumiendo que `xi` es una secuencia
        ordenada e `yi` son los puntos a interpolar. Supone que la
        segunda derivada es continua en cada nodo. En cada segmento
        definido por `xi`, se usa un polinomio de la forma:

        pi(x) = yi + si (x-xi) + ai(x-xi)**2 + bi(x-xi)**3

        de modo que `pi(xi)=yi` y `pi'(xi)=si`. Los coeficientes `si`
        se encuentran imponiendo que `pi(x[i+1])=y[i+1]` y
        `pi'(x[i+1])=s[i+1]`. Los coeficientes `ai` y `bi` se calculan
        usando la clase madre `CSpline`.

        Argumentos
        ----------
        xi : (iterable) Lista ordenada de puntos, correspondientes a los nodos.
        yi : (iterable) Puntos a interpolar.

        Ejemplo
        -------
        p = CSpline_Clasica(xi, yi)  # calcula coeficientes usando xi, yi
        p(x)                         # evalua en un cierto valor de x

        """

        dx = np.diff(xi)
        dy = np.diff(yi)

        # Supone que la segunda derivada es continua, lo que permite
        # encontrar las primeras derivadas `si` mediante la ecuación
        # matricial `A s = p` con `A` una matriz tridiagonal.
        # Esta matriz solo guarda los 3 elementos diagonales
        A = np.zeros((3,dx.size+1))

        # diagonal superior
        A[0,1] = 1.0
        A[0,2:] = dx[:-1]

        # diagonal principal
        A[1,0]  = 2.0
        A[1,1:-1] = 2.0 * (dx[:-1] + dx[1:])
        A[1,-1] = 2.0

        # diagonal inferior
        A[2,-2] = 1.0
        A[2,:-2] = dx[1:]

        # lado derecho de la ecuacion `A si = p`
        p = np.zeros(dx.size+1)

        p[:-1] = 3*dy/dx
        p[-1] = p[-2]
        p[1:-1] = p[1:-1] * dx[:-1] + p[:-2] * dx[1:]

        # encuentra la primera derivada
        si = solve_banded((1,1), A, p)

        # calcula el resto de los coeficientes y hereda otras
        # propiedades de la clase madre.
        super().__init__(xi, yi, si)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def lorentziana(x):
        return 1.0 / (1.0 + 25 * x**2)

    npoints = 10
    xi = np.linspace(-1, 1, npoints)
    yi = lorentziana(xi)        # npoints

    # derivadas al azar
    si = np.random.randn(npoints)

    # metodo usando derivadas manuales
    p = CSpline(xi, yi, si)
    x_eval = np.linspace(-1, 1, 100)

    plt.plot(xi, yi, 'o', label='Nodos')
    plt.plot(x_eval, p(x_eval), label='cSpline con derivadas aleatorias')
    plt.legend()
    plt.show()

    # metodo usando cspline clasica, invirtiendo una matriz
    # tridiagonal. Notar que no usa `si`, este se calcula internamente
    p = CSpline_Clasica(xi, yi)

    plt.plot(xi, yi, 'o', label='Nodos')
    plt.plot(x_eval, p(x_eval), label='cspline clásica')
    plt.legend()
    plt.show()
