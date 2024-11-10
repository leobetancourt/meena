from abc import ABC, abstractmethod
from typing import Union

from jax import Array, lax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.common.helpers import linspace_cells, logspace_cells


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


class Coords:
    CARTESIAN = "cartesian"
    POLAR = "polar"


Prims = Union[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike], 
              tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]]
Cons = Union[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike], 
              tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]]
BoundaryCondition = tuple[str, str]

class Lattice:
    def __init__(self, dims: int, coords: str, 
                 bc_x1: BoundaryCondition = None, bc_x2: BoundaryCondition = None, bc_x3: BoundaryCondition = None, 
                 nx1: int = None, nx2: int = None, nx3: int = None, 
                 x1_range: tuple[float, float] = None, x2_range: tuple[float, float] = None, x3_range: tuple[float, float] = None,
                 num_g: int = 2, 
                 log_x1: bool = False, log_x2: bool = False, log_x3: bool = None):
        self.dims = dims
        self.coords = coords
        self.num_g = num_g

        if nx1:
            self.nx1 = nx1
            self.x1_min, self.x1_max = x1_range
            self.x1, self.x1_intf = logspace_cells(self.x1_min, self.x1_max, num=nx1) \
                                    if log_x1 else linspace_cells(self.x1_min, self.x1_max, num=nx1)
            self.bc_x1 = bc_x1
        if nx2:
            self.nx2 = nx2
            self.x2_min, self.x2_max = x2_range
            self.x2, self.x2_intf = logspace_cells(self.x2_min, self.x2_max, num=nx2) \
                                    if log_x2 else linspace_cells(self.x2_min, self.x2_max, num=nx2)
            self.bc_x2 = bc_x2
        if nx3:
            self.nx3 = nx3
            self.x3_min, self.x3_max = x3_range
            self.x3, self.x3_intf = logspace_cells(self.x3_min, self.x3_max, num=nx3) \
                                    if log_x3 else linspace_cells(self.x3_min, self.x3_max, num=nx3)
            self.bc_x3 = bc_x3

        if self.dims == 1:
            self.X1 = self.x1
        elif self.dims == 2:
            self.X1, self.X2 = jnp.meshgrid(self.x1, self.x2, indexing="ij")
            self.X1_INTF, _ = jnp.meshgrid(self.x1_intf, self.x2, indexing="ij")
            _, self.X2_INTF = jnp.meshgrid(self.x1, self.x2_intf, indexing="ij")
            self.dX1 = self.X1_INTF[1:, :] - self.X1_INTF[:-1, :]
            self.dX2 = self.X2_INTF[:, 1:] - self.X2_INTF[:, :-1]
        elif self.dims == 3:
            self.X1, self.X2, self.X3 = jnp.meshgrid(self.x1, self.x2, self.x3, indexing="ij")
            self.X1_INTF, _, _ = jnp.meshgrid(self.x1_intf, self.x2, self.x3, indexing="ij")
            _, self.X2_INTF, _ = jnp.meshgrid(self.x1, self.x2_intf, self.x3, indexing="ij")
            _, _, self.X3_INTF = jnp.meshgrid(self.x1, self.x2, self.x3_intf, indexing="ij")
            self.dX1 = self.X1_INTF[1:, :, :] - self.X1_INTF[:-1, :, :]
            self.dX2 = self.X2_INTF[:, 1:, :] - self.X2_INTF[:, :-1, :]
            self.dX3 = self.X3_INTF[:, :, 1:] - self.X3_INTF[:, :, :-1]


class Hydro(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            print(key, value)
            setattr(self, key, value)

    @abstractmethod
    def initialize(self, lattice: Lattice) -> Array:
        pass

    @abstractmethod
    def resolution(self) -> Union[int, tuple[int, int], tuple[int, int, int]]:
        pass

    def t_start(self) -> float:
        return 0

    def t_end(self) -> float:
        return 1

    def save_interval(self) -> float:
        return None

    def range(self) -> Union[tuple[float, float], 
                             tuple[tuple[float, float], tuple[float, float]], 
                             tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        return ((0, 1), (0, 1))

    def num_g(self) -> int:
        if self.regime() == "HD":
            return 2
        else:
            return 3

    def log_x1(self) -> bool:
        return False

    def log_x2(self) -> bool:
        return False
    
    def log_x3(self) -> bool:
        return False
    
    def time_order(self) -> int:
        return 1
    
    def solver(self) -> str:
        return "hll"

    def PLM(self) -> bool:
        return False
    
    def theta_PLM(self) -> float:
        return 1.5

    def cfl(self) -> float:
        return 0.4

    def timestep(self) -> float:
        return None

    def coords(self) -> str:
        return Coords.CARTESIAN

    def bc_x1(self) -> BoundaryCondition:
        return (Boundary.OUTFLOW, Boundary.OUTFLOW)

    def bc_x2(self) -> BoundaryCondition:
        return (Boundary.OUTFLOW, Boundary.OUTFLOW)
    
    def bc_x3(self) -> BoundaryCondition:
        return (Boundary.OUTFLOW, Boundary.OUTFLOW)

    def gamma(self) -> float:
        return 5/3

    def nu(self) -> float:
        return None
    
    def regime(self) -> str:
        return "HD"
    
    def regime_index(self) -> str:
        return 0 if self.regime() == "HD" else 1
    
    def P(self, cons: Cons, *args) -> Array:
        rho = cons[0]
        u, v = cons[1] / rho, cons[2] / rho
        e = cons[3]
        return (self.gamma() - 1) * (e - (0.5 * rho * (u ** 2 + v ** 2)))
    
    def E(self, prims: Prims, *args) -> Array:
        rho, u, v, p = prims
        return (p / (self.gamma() - 1)) + (0.5 * rho * (u ** 2 + v ** 2))
        
    def c_s(self, prims: Prims, *args) -> Array:
        rho, _, _, p = prims
        return jnp.sqrt(self.gamma() * p / rho)
      
    def source(self, U: ArrayLike, *args) -> Array:
        return jnp.zeros_like(U)
    
    def self_gravity(self) -> bool:
        return False
    
    def G(self) -> float:
        return 1

    def check_U(self, lattice: Lattice, U: ArrayLike, t: float) -> Array:
        return U

    def diagnostics(self):
        return []
