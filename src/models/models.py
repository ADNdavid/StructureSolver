from enum import Enum

from pydantic import BaseModel, computed_field, ConfigDict, model_validator
from sympy.matrices import Matrix
from sympy import symbols, pprint, cos, sin, asin, deg, simplify, factor, rad
import numpy as np

class StiffnessMatrix():
    A, E, I, L, theta = symbols('A E I L theta')
    # Matriz simbólica de rigidez local para un elemento sometido a fuerza axial, fuerzas cortante y momento. La matriz se basa en la teoría de vigas de Euler-Bernoulli.
    symbolic_local_stiffness_matrix = Matrix([   
                [A*E/L, 0, 0, -A*E/L, 0, 0],
                [0, 12*E*I/L**3, 6*E*I/L**2, 0, -12*E*I/L**3, 6*E*I/L**2],
                [0, 6*E*I/L**2, 4*E*I/L, 0, -6*E*I/L**2, 2*E*I/L],
                [-A*E/L, 0, 0, A*E/L, 0, 0],
                [0, -12*E*I/L**3, -6*E*I/L**2, 0, 12*E*I/L**3, -6*E*I/L**2],
                [0, 6*E*I/L**2, 2*E*I/L, 0, -6*E*I/L**2, 4*E*I/L]
    ])
    
    # Matriz de transformación para un elemento con una orientación dada por el ángulo theta. Esta matriz se utiliza para transformar las coordenadas locales del elemento a coordenadas globales, teniendo en cuenta la rotación del elemento en el espacio.
    symbolic_transformation_matrix = Matrix([
                [cos(theta), sin(theta), 0, 0, 0, 0],
                [-sin(theta), cos(theta), 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cos(theta), sin(theta), 0],
                [0, 0, 0, -sin(theta), cos(theta), 0],
                [0, 0, 0, 0, 0, 1]
    ])
    
    symbolic_global_stiffness_matrix = symbolic_transformation_matrix.T * symbolic_local_stiffness_matrix * symbolic_transformation_matrix
    
    def __init__(self, a: float, e: float, i: float, l: float, theta: float):
        self.global_stiffness_matrix = self.symbolic_global_stiffness_matrix.subs({self.A: a, self.E: e, self.I: i, self.L: l, self.theta: theta})
        
    def print_matrix(self):
        print("Matriz calculada de rigidez del elemento [K] (Coordenadas globales):")
        pprint(self.global_stiffness_matrix)

    def print_symbolic_matrix(self):
        print("Matriz simbólica [K] (Coordenadas locales):")
        pprint(self.symbolic_local_stiffness_matrix)
        print("Matriz simbólica de transformación [T]:")
        pprint(self.symbolic_transformation_matrix)
        print("Matriz simbólica [K] (Coordenadas globales) = [T]**t.[K-local].[T]:")
        pprint(self.symbolic_global_stiffness_matrix)

class EnumTypesOfDegreeOfFreedom(str, Enum):
    X_TRANSLATION = "x_translation"
    Y_TRANSLATION = "y_translation"
    Z_ROTATION = "z_rotation"

class DegreeOfFreedom(BaseModel):
    tag_number: int = 0
    type_of_degree_of_freedom: EnumTypesOfDegreeOfFreedom
    constrained: bool = False
    model_config = ConfigDict(frozen=False)


class Joint(BaseModel):
    tag: str
    x_coordinate: float
    y_coordinate: float
    degrees_of_freedom: list[DegreeOfFreedom] = []
    model_config = ConfigDict(frozen=False)
    
    @model_validator(mode="after")
    def initialize_degrees_of_freedom(self):
        self.degrees_of_freedom = [
            DegreeOfFreedom(type_of_degree_of_freedom=EnumTypesOfDegreeOfFreedom.X_TRANSLATION),
            DegreeOfFreedom(type_of_degree_of_freedom=EnumTypesOfDegreeOfFreedom.Y_TRANSLATION),
            DegreeOfFreedom(type_of_degree_of_freedom=EnumTypesOfDegreeOfFreedom.Z_ROTATION)
        ]
        return self


class CrossSection(BaseModel):
    A: float
    I: float


class Element(BaseModel):
    tag: str
    initial_joint: Joint
    final_joint: Joint
    cross_section: CrossSection
    elastic_modulus: float # Modulo de elasticidad o de Young (E)
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False) # Provisional, para permitir el uso de objetos de tipo StiffnessMatrix en la clase Element sin que Pydantic lance un error de validación. Esto es necesario porque la matriz de rigidez se calcula a partir de las propiedades del elemento y no es un campo que se defina directamente al crear una instancia de Element.
    
    @computed_field
    @property
    def length(self) -> float:
        dx = self.final_joint.x_coordinate - self.initial_joint.x_coordinate
        dy = self.final_joint.y_coordinate - self.initial_joint.y_coordinate
        return (dx**2 + dy**2)**0.5
    
    @computed_field
    @property
    def inclination_angle(self) -> float: # El ángulo de inclinación del elemento con respecto al eje horizontal
        dx = self.final_joint.x_coordinate - self.initial_joint.x_coordinate
        dy = self.final_joint.y_coordinate - self.initial_joint.y_coordinate
        return float(asin(dy / (dx**2 + dy**2)**0.5))

    @computed_field
    @property
    def stiffness_matrix(self) -> StiffnessMatrix:
        return StiffnessMatrix(
            a=self.cross_section.A, 
            e=self.elastic_modulus, 
            i=self.cross_section.I, 
            l=self.length, 
            theta=self.inclination_angle
        )


# Carga puntual
class PointLoad(BaseModel):
    magnitude: float
    length_position: float
    direction: float = 90.0 # El ángulo de la dirección de la carga con respecto al eje local del elemento

# Carga distribuida
class DistributedLoad(BaseModel):
    initial_magnitude: float
    final_magnitude: float
    initial_length_position: float # La posición a lo largo del elemento donde comienza la carga, medida desde el nodo inicial del elemento
    final_length_position: float # La posición a lo largo del elemento donde termina la carga, medida desde la posición inicial de la carga
    direction: float = 90.0 # El ángulo de la dirección de la carga con respecto al eje local del elemento

# Carga de momento
class MomentLoad(BaseModel):
    magnitude: float
    length_position: float # La posición a lo largo del elemento donde se aplica el momento, medida desde el nodo inicial del elemento

class EnumSupportType(str, Enum):
    FIXED  = "fixed"  # Empotrado
    PINNED = "pinned" # Articulado
    ROLLER = "roller" # Rodillo

class Support(BaseModel):
    tag: str
    joint: Joint
    support_type: EnumSupportType
    inclination_angle: float = 0.0

# Fuerzas de Fijación y Momentos de empotramiento
class FixedEndMoment():    
    def __init__(self, element: Element, load: PointLoad | DistributedLoad | MomentLoad):
        self.element = element
        self.load = load
        self._local_forces = None
        self._global_forces = None
    
    def get_symbolic_functions(self):
        a, b, F, L, M, Wi, Wf, alpha  = symbols('a b F L M Wi Wf alpha')
        # Reacciones Ri, Rj; Momentos Mi, Mj
        rix, rjx, riy, rjy, miz, mjz = 0, 0, 0, 0, 0, 0    
        if isinstance(self.load, PointLoad):
            # Carga Puntual
            rix = (F * cos(alpha)) * b / L
            rjx = (F * cos(alpha)) * a / L
            riy = (F * sin(alpha)) * ((b ** 2) / (L ** 2)) * (3 - 2 * (b / L))
            rjy = (F * sin(alpha)) * ((a ** 2) / (L ** 2)) * (3 - 2 * (a / L))
            miz = ((F * sin(alpha)) * a * (b ** 2)) / (L ** 2)
            mjz = -((F * sin(alpha)) * (a ** 2) * b ) / (L ** 2) #! consultar esta formula, ya que en algunas fuentes el momento es (-) y en otros (+) 
        elif isinstance(self.load, DistributedLoad):
            # Carga Trapezoidal
            riy = b * (Wf * (10 * L **3 - 15 *L * b **2 + 20 * a ** 3 + a ** 2 * (-30 * L + 40 * b) - 10 * a * b * (4 * L - 3 * b) + 8 * b **3 ) + Wi * (10 * L ** 3 - 5 * L * b ** 2 + 20 * a ** 3 + a ** 2 * (-30 * L + 20 * b) - 10 * a * b * (2 * L - b) + 2 * b ** 3)) / (20 * L ** 3) * sin(alpha)
            rjy = - b * (-Wf * (15 * L * b **2 - 20 * a ** 3 + a ** 2 * (30 * L - 40 * b) + 10 * a * b * (4 * L - 3 * b) - 8 * b ** 3) - Wi * (5 * L * b ** 2 - 20 * a ** 3 + a ** 2 * (30 * L - 20 * b) + 10 * a * b * (2 * L - b) - 2 * b ** 3)) / (20 * L ** 3) * sin(alpha)
            miz = b * (Wf * (20 * L ** 2 * b - 30 * L * b ** 2 + 30 * a ** 3 + 60 * a ** 2 * (-L + b) + 5 * a * (6 * L ** 2 - 16 * L * b + 9 * b ** 2) + 12 * b ** 3) + Wi * (10 * L ** 2 * b - 10 * L *b ** 2 + 30 * a ** 3 + 30 * a ** 2 *(-2 * L + b) + 5 * a * (6 * L ** 2 - 8 * L * b + 3 * b ** 2) + 3 * b ** 3)) / (60 * L ** 2) * sin(alpha)
            mjz = - b * (Wf * (15 * L * b ** 2 - 30 * a ** 3 + 30 * a ** 2 * (L - 2 * b) + 5 * a * b * (8 * L - 9 * b) - 12 * b ** 3) + Wi * (5 * L * b ** 2 - 30 * a ** 3 + 30 * a ** 2 * (L - b) + 5 * a * b * (4 * L - 3 * b) - 3 * b ** 3)) / (60 * L ** 2) * sin(alpha) #! consultar esta formula, ya que en algunas fuentes el momento es (-) y en otros (+)
            rix = riy * (cos(alpha) / sin(alpha))
            rjx = rjy * (cos(alpha) / sin(alpha))
        elif isinstance(self.load, MomentLoad):
            # Carga momento
            riy = M * ((6 * a * b) / (L ** 3))
            rjy = - M * ((6 * a * b) / (L ** 3))
            miz = M * (b / L) * (2 - 3 * (b / L))
            mjz = M * (a / L) * (2 - 3 * (a / L))
        return rix, riy, miz, rjx, rjy, mjz
    
    def print_functions(self):
        rix, riy, miz, rjx, rjy, mjz = self.get_symbolic_functions()
        print("Funciones para calcular las reacciones y momentos de empotramiento perfecto:")
        print("Reacción Rix:")
        pprint(simplify(rix))
        print("Reacción Riy:")
        pprint(simplify(riy))
        print("Momento Miz:")
        pprint(simplify(miz))
        print("Reacción Rjx:")
        pprint(simplify(rjx))
        print("Reacción Rjy:")
        pprint(simplify(rjy))
        print("Momento Mjz:")
        pprint(simplify(mjz))

    def calculate_functions(self):
        a, b, F, L, M, Wi, Wf, theta, alpha  = symbols('a b F L M Wi Wf theta alpha')
        functions = Matrix(self.get_symbolic_functions())
        
        function_values = {L:self.element.length}
        if isinstance(self.load, PointLoad): 
            function_values.update({F: self.load.magnitude, 
                                    a: self.load.length_position, 
                                    b: self.element.length - self.load.length_position, 
                                    alpha: rad(self.load.direction)})
        elif isinstance(self.load, DistributedLoad):        
            function_values.update({Wi: self.load.initial_magnitude, 
                                    Wf: self.load.final_magnitude, 
                                    a: self.load.initial_length_position, 
                                    b: self.load.final_length_position, 
                                    alpha: rad(self.load.direction)})
        elif isinstance(self.load, MomentLoad):
            function_values.update({M: self.load.magnitude, 
                                    a: self.load.length_position, 
                                    b: self.element.length - self.load.length_position})
        
        print(f"Resultados elemento ({self.element.tag})")
        print(f"longitud: {self.element.length}, angulo de inclinación (rad): {self.element.inclination_angle}, (grados):{float(deg(self.element.inclination_angle))}")
        self._local_forces = functions.subs(function_values).evalf()
        print("fuerzas locales:")
        pprint(self._local_forces)
        transformation_matrix = self.element.stiffness_matrix.symbolic_transformation_matrix

        print("Momentos de empotramiento perfecto:")
        self._global_forces = (transformation_matrix.T * self._local_forces).subs({theta: self.element.inclination_angle}).evalf()
        print("Coordenadas globales:")
        pprint(self._global_forces)

    def get_local_forces(self):
        if self._local_forces is None:
            self.calculate_functions()
        return self._local_forces

    def get_global_forces(self):
        if self._global_forces is None:
            self.calculate_functions()
        return self._global_forces

class Structure(BaseModel):
    joints: list[Joint]
    elements: list[Element]
    supports: list[Support]
    loads: list[FixedEndMoment]
    model_config = ConfigDict(arbitrary_types_allowed=True)
