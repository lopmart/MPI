# Descripción: Programa que suma dos vectores de tamaño N de forma paralela utilizando MPI.
%%writefile suma_vectores.py
from mpi4py import MPI
import numpy as np

# Inicializar el entorno MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rango del proceso
size = comm.Get_size()  # Número total de procesos

# Tamaño de los vectores a sumar
N = 100

# Crear los vectores A y B solo en el proceso raíz
if rank == 0:
    A = np.random.rand(N)  # Vector A con números aleatorios
    B = np.random.rand(N)  # Vector B con números aleatorios
else:
    A = None
    B = None

# Calcular el tamaño del "bloque" que cada proceso manejará
local_size = N // size

# Crear subvectores para cada proceso
local_A = np.empty(local_size, dtype='d')
local_B = np.empty(local_size, dtype='d')

start = MPI.Wtime()
# Distribuir partes del vector A y B a cada proceso
comm.Scatter(A, local_A, root=0)
comm.Scatter(B, local_B, root=0)

# Realizar la suma local de los vectores
local_C = local_A + local_B


# Recoger todos los subresultados en el proceso raíz
if rank == 0:
    C = np.empty(N, dtype='d')  # Vector resultado en el proceso raíz
else:
    C = None

comm.Gather(local_C, C, root=0)
end = MPI.Wtime()


# Mostrar el resultado en el proceso raíz
if rank == 0:
    print(f"Tiempo de reducción: {end - start} segundos")
    print("Suma de los dos vectores completada.")
    print(C)  # vector resultante
