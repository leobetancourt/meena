from abc import ABC, abstractmethod

from jax import Array
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


Primitives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
Conservatives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
BoundaryCondition = tuple[str, str]


class Lattice:
    def __init__(self, coords: str, bc_x1: BoundaryCondition, bc_x2: BoundaryCondition, nx1: int, nx2: int, x1_range: tuple[float, float], x2_range: tuple[float, float], num_g: int = 2, log_x1: bool = False, log_x2: bool = False):
        self.coords = coords
        self.num_g = num_g
        self.bc_x1 = bc_x1
        self.bc_x2 = bc_x2
        self.nx1, self.nx2 = nx1, nx2
        self.x1_min, self.x1_max = x1_range
        self.x2_min, self.x2_max = x2_range

        if log_x1:
            self.x1, self.x1_intf = logspace_cells(
                self.x1_min, self.x1_max, num=nx1)
        else:
            self.x1, self.x1_intf = linspace_cells(
                self.x1_min, self.x1_max, num=nx1)
        if log_x2:
            self.x2, self.x2_intf = logspace_cells(
                self.x2_min, self.x2_max, num=nx2)
        else:
            self.x2, self.x2_intf = linspace_cells(
                self.x2_min, self.x2_max, num=nx2)
        self.X1, self.X2 = jnp.meshgrid(self.x1, self.x2, indexing="ij")
        self.X1_INTF, _ = jnp.meshgrid(self.x1_intf, self.x2, indexing="ij")
        _, self.X2_INTF = jnp.meshgrid(self.x1, self.x2_intf, indexing="ij")
        self.dX1 = self.X1_INTF[1:, :] - self.X1_INTF[:-1, :]
        self.dX2 = self.X2_INTF[:, 1:] - self.X2_INTF[:, :-1]


class Hydro(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            print(key, value)
            setattr(self, key, value)

    @abstractmethod
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        pass

    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        pass

    def t_start(self) -> float:
        return 0

    def t_end(self) -> float:
        return 1

    def save_interval(self) -> float:
        return None

    def regime(self) -> str:
        return "HD"

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0, 1), (0, 1))

    def num_g(self) -> int:
        return 2

    def log_x1(self) -> bool:
        return False

    def log_x2(self) -> bool:
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

    def gamma(self) -> float:
        return 5.0 / 3.0

    def nu(self) -> float:
        return None

    def E(self, prims: Primitives, X1: ArrayLike = None, X2: ArrayLike = None, t: float = None) -> Array:
        rho, u, v, p = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
        return p / (self.gamma() - 1) + 0.5 * rho * (u ** 2 + v ** 2)

    def c_s(self, prims: Primitives, X1: ArrayLike = None, X2: ArrayLike = None, t: float = None) -> Array:
        rho, p = prims[..., 0], prims[..., 3]
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike = None, X2: ArrayLike = None, t: float = None) -> Array:
        rho, mom_x, mom_y, e = cons[..., 0], cons[..., 1], cons[..., 2], cons[..., 3]
        u, v = mom_x / rho, mom_y / rho
        return (self.gamma() - 1) * (e - (0.5 * rho * (u ** 2 + v ** 2)))

    def source(self, prims: ArrayLike, X1: ArrayLike = None, X2: ArrayLike = None, t: float = None) -> Array:
        return jnp.zeros_like(prims)
    
    def self_gravity(self) -> bool:
        return False
    
    def G(self) -> float:
        return 1

    def check_U(self, lattice: Lattice, U: ArrayLike, t: float) -> Array:
        return U

    def diagnostics(self):
        return []
