#!/bin/python

__all__ = ["SaltoRana"]

import numpy as np


def SaltoRana(a, r0, v0, t0, dt=0.01, steps=100, **kwargs):
    """Usa el método del salto de la rana para resolver ecuaciones diferenciales de la forma `r''(t) = a(r(t),t)`, o bien
    .. math::
        r'(t) = v(t)
        v'(t) = a(r(t), t)

    con condiciones iniciales `r(t0)=r0` y `v(t0)=v0`. La integración se realiza entre los tiempos `t=t0` y `t=steps*dt`, con `dt` el tamaño del paso de tiempo. Aquí, `r` (tipicamente la posicion) y `v` (tipicamente la velocidad) pueden ser cantidades vectoriales, y `a=a(r,t)` es una función de al menos dos variables, la posición y el tiempo. Cualquier argumento extra será pasado a `a(r,t)`.

    Argumentos
    ----------
    a : Funcion de al menos dos argumentos, `a(r,t, ...)`.
    r0, v0, t0 : Condiciones inicial `r(t0)=r0` y `v(t0)=v0` a tiempo `t=t0`.
    dt : (opcional) Tamaño del paso de tiempo.
    steps : (opcional) Número de pasos de tiempo.
    **kwargs : Otros argumentos que serán pasados a la función `a`.

    Retorna
    -------
    t : Tiempo `t=t0+n*dt` con `0 <= n < steps`.
    r : Solución para la posición para todos los instantes de tiempo
    v : Solución para la velocidad para todos los instantes de tiempo
    """

    # lista de tiempos (asume monotomicamente creciente)
    t = t0 + np.arange(steps) * dt

    # crea dos listas vacias de `steps` elementos
    r = np.zeros(steps)
    v = np.zeros(steps)

    # inicializa usando las condiciones iniciales
    r[0], v[0] = r0, v0

    # evalua aceleracion inicial
    a0 = a(r0, t0, **kwargs)

    # usa el metodo del salto de la rana en cada paso de tiempo
    for n in range(steps-1):
        vmedio = v[n] + 0.5*dt*a0

        r[n+1] = r[n] + dt * vmedio

        a0 = a(r[n+1], t[n+1], **kwargs)    	# reusa a0 a tiempo t[n+1]

        v[n+1] = vmedio + 0.5*dt*a0

    return t, r, v



# Esta parte permite mostrar un código de ejemplo del módulo
if __name__ == "__main__":
   import matplotlib.pyplot as plt

   def a(r, t, w=1.0):
       """Aceleración para el oscilador armónico simple, r''(t)+w**2 * sin(r)==0, donde `w` es la frecuencia de oscilación. Note que esta ecuación no depende explícitamente del tiempo."""

       return -np.sin(r) * w**2


   # Resuelve ecuación de oscilador armónico.
   # Notar que usamos `w=2.0` en vez de `**kwargs`.
   t, r, v = SaltoRana(a, r0=2.0, v0=0.0, t0=0.0, dt=0.01, steps=1_000, w=1.0)

   # Graficamos
   plt.plot(t, r, label="$r(t)$")
   plt.plot(t, v, label="$v(t)$")
   plt.legend()

   plt.xlabel("tiempo")
   plt.ylabel("solución")
   plt.show()

