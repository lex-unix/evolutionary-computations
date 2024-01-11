from src.lib.functions import Ackley
from src.lib.functions import Easom
from src.lib.functions import Sphere
from src.lib.functions import Sphere3D
from src.lib.functions import ThreeHumpCamel

FUNCTIONS_1D = [Sphere()]
FUNCTIONS_2D = [ThreeHumpCamel(), Ackley(), Easom()]
FUNCTIONS_ND = [Sphere3D()]
