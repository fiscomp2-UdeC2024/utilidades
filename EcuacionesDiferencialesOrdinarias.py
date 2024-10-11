#!/bin/python

__all__ = ["SaltoRana", "RungeKutta4", "Euler", "EulerCromer"]

import numpy as np


def _condiciones_iniciales(steps, *args):
    """Función conveniente para configurar condiciones iniciales.

    Argumentos
    ----------
    steps : (entero) Número de pasos de tiempo luego de dar las condiciones iniciales.
    r0, v0, ... : (argumentos opcionales) Condiciones iniciales para un número arbitrario de variables.

    Retorna
    -------
    arrays : (iterable) Un arreglo de funciones de tamaño (N, steps, casos), donde `N` es la cantidad de variables que necesitan condiciones iniciales, `steps` el número de steps para resolver una ecuacion diferencial, y casos es el tamaño de cada condicion inicial.
    """
    shape = [np.asarray(r0).shape for r0 in args]

    if len({*shape})>1:
        raise ValueError(f"Condiciones iniciales no tienen el mismo tamaño, con shapes: {shape}")

    shape = shape[0]
    arrays = []
    for i,r0 in enumerate(args):
        r = np.zeros((steps, *shape))
        r[0] = r0
        arrays.append(r)

    return arrays[0] if len(arrays) == 1 else arrays


def SaltoRana(a, r0, v0, t, **kwargs):
    """Método del salto de la rana para resolver ecuaciones de la forma r'(t) = v(t) y v'(t) = a(r(t), t). Si se conocen r(t) y v(t), es metodo retorna
        v(t+dt/2) = v(t-dt/2) + dt * a(r(t), t)
        r(t+dt) = r(t) + dt * v(t+dt/2)

    La integración se realiza para todos los tiempos definidos en el iterable `t`.

    Argumentos
    ----------
    a      : (función) Funcion de al menos dos argumentos, `a(r,t, ...)`.
    r0, v0 : (escalar o iterable) Condiciones iniciales `r(t[0])=r0` y `v(t[0])=v0` a tiempo `t[0]`.
    t      : (iterable) Lista de tiempos donde se realiza la integración.
    **kwargs : Otros argumentos que serán pasados a la función `a`.

    Retorna
    -------
    r : (iterable) Solución para la posición para cada instante de tiempo. Su dimensión es (t.size, *r0.shape)
    v : (iterable) Solución para la velocidad para cada instante de tiempo. Su dimensión es (t.size, *r0.shape)
    """

    dt = np.diff(t)
    r, v = _condiciones_iniciales(t.size, r0, v0)

    # evalua aceleracion inicial
    a0 = a(r0, t[0], **kwargs)

    # usa el metodo del salto de la rana en cada paso de tiempo
    for n in range(t.size-1):
        vmedio = v[n] + 0.5*dt[n]*a0

        r[n+1] = r[n] + dt[n] * vmedio

        a0 = a(r[n+1], t[n+1], **kwargs)    	# reusa a0 a tiempo t[n+1]

        v[n+1] = vmedio + 0.5*dt[n]*a0

    return r, v


def Euler(f, r0, t, **kwargs):
    """Método de Euler para ecuaciones de la forma r'(t) = f(r, t). Si se conoce r(t), este metodo retorna

        r(t+dt) = r(t) + dt * f(r(t), t)

    Si f depende de parametros, estos son ingresados mediante **kwargs.

    Argumentos
    ----------
    f  : (función) Función de al menos dos argumentos, `f(r,t, ...)`.
    r0 : (escalar o iterable) Condicion inicial `r(t0)=r0` a tiempo `t=t0`.
    t  : (iterable) Lista de tiempos.
    **kwargs : Otros argumentos que serán pasados a la función `f`.

    Retorna
    -------
    r : Solución para cada instante de tiempo `t`. Es un arreglo de tamaño (t.size, r0.shape)
    """

    dt = np.diff(t)
    r = _condiciones_iniciales(t.size, r0)

    # usa el metodo de Euler en cada paso de tiempo
    for n in range(t.size-1):
        r[n+1] = r[n] + dt[n] * f(r[n], t[n], **kwargs)

    return r


def EulerCromer(a, r0, v0, t, **kwargs):
    """Método de Euler-Cromer para ecuaciones de la forma r'(t) = v(t) y v'(t) = a(r(t),t). Si se conoce r(t) y v(t), este metodo retorna

        r(t+dt) = r(t) + dt * v(t)
        v(t+dt) = v(t) + dt * a(r(t+dt), t+dt)

    Argumentos
    ----------
    a      : (función) Funcion de al menos dos argumentos, `a(r,t, ...)`.
    r0, v0 : (escalar o iterable) Condiciones iniciales `r(t0)=r0` y `v(t0)=v0` a tiempo `t0=t[0]`.
    t      : (iterable) Lista de tiempos donde se realiza la integración.
    **kwargs : Otros argumentos que serán pasados a la función `a`.

    Retorna
    -------
    r : (iterable) Posición para cada tiempo t y cada posición inicial r0, shape=(t.size, *r0.shape)
    v : (iterable) Velocidad para cada tiempo t y cada posición inicial v0, shape=(t.size, *v0.shape)

    """

    dt = np.diff(t)
    r, v = _condiciones_iniciales(t.shape, r0, v0)

    # Metodo de Euler-Cromer
    for n in range(t.size-1):
        r[n+1] = r[n] + dt[n] * v[n]
        v[n+1] = v[n] + dt[n] * a(r[n+1], t[n+1], **kwargs)

    return r, v


def RungeKutta4(f, r0, t, *args, **kwargs):
    """Método de Runge-Kutta de cuarto orden para ecuaciones de la forma r'(t) = f(r,t). Si se conoce r(t), este metodo retorna

        r(t+dt) = r(t) + (dt/6) * (K1 + 2 K2 + 2 K3 + K4)

    donde

        K1 = f(r(t), t)
        K2 = f(r(t) + K1 dt/2, t+dt/2)
        K3 = f(r(t) + K2 dt/2, t+dt/2)
        K4 = f(r(t) + K3 dt, t+dt)

    Argumentos
    ----------
    f  : (función) Función de al menos dos argumentos, `f(r,t, ...)`.
    r0 : (escalar o iterable) Condicion inicial `r(t0)=r0` a tiempo `t=t0`.
    t  : (iterable) Lista de tiempos.
    **kwargs : Otros argumentos que serán pasados a la función `f`.

    Retorna
    -------
    r : Solución para cada instante de tiempo `t`. Es un arreglo de tamaño (t.size, r0.shape)
    """

    dt = np.diff(t)
    r = _condiciones_iniciales(t.size, r0)

    # usa el metodo de Runge-Kutta de 4to orden tradicional
    for n in range(t.size-1):
        K1 = dt[n] * f(r[n]         , t[n]          , **kwargs)
        K2 = dt[n] * f(r[n] + 0.5*K1, t[n]+0.5*dt[n], **kwargs)
        K3 = dt[n] * f(r[n] + 0.5*K2, t[n]+0.5*dt[n], **kwargs)
        K4 = dt[n] * f(r[n] + K3    , t[n]+dt[n]    , **kwargs)

        r[n+1] = r[n] + (K1 + 2 * K2 + 2 * K3 + K4) / 6.0

    return r




# Esta parte permite mostrar un código de ejemplo del módulo
if __name__ == "__main__":
   import matplotlib.pyplot as plt

   def a(r, t, w=1.0):
       """Aceleración para el oscilador armónico simple, r''(t)+w**2 * sin(r)==0, donde `w` es la frecuencia de oscilación. Note que esta ecuación no depende explícitamente del tiempo."""

       return -np.sin(r) * w**2

   # Aqui probamos rutinas como SaltoRana o EulerCromer que tienen la misma estructura
   # define lista de tiempos entre 0<= t <= 20 dividido en 256 intervalos
   t = np.linspace(0, 20, 256)

   # condiciones iniciales (notar que es una lista de condiciones iniciales)
   r0 = np.linspace(0,6.0,10)
   v0 = np.zeros(*r0.shape)

   # Resuelve ecuación de oscilador armónico.
   # Notar que usamos `w=algo` en vez de `**kwargs`.
   r, v = SaltoRana(a, r0=r0, v0=v0, t=t, w=1.0)

   # Graficamos
   fig, ax = plt.subplots(2, sharex=True)
   handles = ax[0].plot(t, r)
   ax[1].plot(t, v)
   ax[0].legend(handles=handles, labels=[f"r0={r0:.2f} v0={v0:.2f}" for r0,v0 in zip(r[0],v[0])])

   ax[1].set_xlabel("tiempo")
   ax[0].set_ylabel("posición")
   ax[1].set_ylabel("velocidad")
   plt.show()


   # Aqui probamos rutinas como Euler o RungeKutta4
   def f(x, t):
       return np.array([x[1], -np.sin(x[0])])

   x = RungeKutta4(f, r0=(r0,v0), t=t)
   r = x[:,0]
   v = x[:,1]