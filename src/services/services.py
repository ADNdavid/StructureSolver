from ..models import models
import numpy as np
import sympy as sp
import pandas as pd

class StructureSolverService:
    def __init__(self):
        self.structure = models.Structure(joints=[], elements=[], supports=[], loads=[])
        self.global_stiffness_matrix = None
        self._count_dof = 0
    
    def add_joint_to_structure(self, x: float, y: float, name: str = "Jx") -> None:
        joint = models.Joint(tag=name, x_coordinate=x, y_coordinate=y)
        self.structure.joints.append(joint)
    
    def delete_joint_from_structure(self, tag: str) -> None:
        self.structure.joints = [joint for joint in self.structure.joints if joint.tag != tag]
    
    def add_element_to_structure(self, initial_joint_tag: str, final_joint_tag: str, cross_section: models.CrossSection, elastic_modulus: float, name: str = "Ex") -> None:
        initial_joint = next((joint for joint in self.structure.joints if joint.tag == initial_joint_tag), None)
        final_joint = next((joint for joint in self.structure.joints if joint.tag == final_joint_tag), None)
        
        if not initial_joint or not final_joint:
            raise ValueError("Initial and final joints must exist in the structure.")
        
        if initial_joint.tag == final_joint.tag:
            raise ValueError("Initial and final joints cannot be the same.")
        
        element = models.Element(
            tag=name,
            initial_joint=initial_joint,
            final_joint=final_joint,
            cross_section=cross_section,
            elastic_modulus=elastic_modulus
        )
        self.structure.elements.append(element)
    
    def delete_element_from_structure(self, tag: str) -> None:
        self.structure.elements = [element for element in self.structure.elements if element.tag != tag]
    
    def add_support_to_structure(self, joint_tag: str, support_type: models.EnumSupportType, inclination_angle: float = 0.0, name: str = "Sx") -> None:
        joint = next((joint for joint in self.structure.joints if joint.tag == joint_tag), None)
        
        if not joint:
            raise ValueError("Joint must exist in the structure.")
        
        support = models.Support(
            tag=name,
            joint=joint,
            support_type=support_type,
            inclination_angle=inclination_angle
        )
        self.structure.supports.append(support)
    
    def delete_support_from_structure(self, tag: str) -> None:
        self.structure.supports = [support for support in self.structure.supports if support.tag != tag]
    
    def _add_load_to_structure(self, element_tag: str, load: models.PointLoad | models.DistributedLoad | models.MomentLoad) -> None:
        element = next((element for element in self.structure.elements if element.tag == element_tag), None)
        
        if not element:
            raise ValueError("Element must exist in the structure.")
        
        equivalent_load = models.FixedEndMoment(element, load)
        self.structure.loads.append(equivalent_load)
    
    def add_puntual_load_to_element(self, element_tag: str, magnitude: float, length_position: float, direction: float = 90.0) -> None:
        load = models.PointLoad(
            magnitude=magnitude,
            length_position=length_position,
            direction=direction
        )
    
        self._add_load_to_structure(element_tag, load)
    
    def add_distributed_load_to_element(self, element_tag: str, initial_magnitude: float, final_magnitude: float, initial_length_position: float, final_length_position: float, direction: float = 90.0) -> None:
        load = models.DistributedLoad(
            initial_magnitude=initial_magnitude, 
            final_magnitude=final_magnitude, 
            initial_length_position=initial_length_position, 
            final_length_position=final_length_position, 
            direction=direction
        )
    
        self._add_load_to_structure(element_tag, load)
    
    def add_puntual_moment_load_to_element(self, element_tag: str, magnitude: float, length_position: float) -> None:
        load = models.MomentLoad(
            magnitude=magnitude,
            length_position=length_position,
        )
    
        self._add_load_to_structure(element_tag, load)
    
    def _constrain_degrees_of_freedom_from_support(self, support_tag: str) -> None:
        support = next((support for support in self.structure.supports if support.tag == support_tag), None)
        
        if not support:
            raise ValueError("Support must exist in the structure.")
        
        for dof in support.joint.degrees_of_freedom:
            if support.support_type == models.EnumSupportType.FIXED and dof.type_of_degree_of_freedom in [
                models.EnumTypesOfDegreeOfFreedom.X_TRANSLATION, 
                models.EnumTypesOfDegreeOfFreedom.Y_TRANSLATION, 
                models.EnumTypesOfDegreeOfFreedom.Z_ROTATION
                ]:
                dof.constrained = True
                #print(f"Restringiendo nodo: {support_tag}, Empotramiento en {dof.type_of_degree_of_freedom}, restringido?: {dof.constrained}, Tag-number: {dof.tag_number}")
            elif support.support_type == models.EnumSupportType.PINNED and dof.type_of_degree_of_freedom in [
                models.EnumTypesOfDegreeOfFreedom.X_TRANSLATION, 
                models.EnumTypesOfDegreeOfFreedom.Y_TRANSLATION
                ]:
                dof.constrained = True
                #print(f"Restringiendo nodo: {support_tag}, Articulacion en {dof.type_of_degree_of_freedom}, restringido?: {dof.constrained}, Tag-number: {dof.tag_number}")
            elif support.support_type == models.EnumSupportType.ROLLER and dof.type_of_degree_of_freedom in [
                models.EnumTypesOfDegreeOfFreedom.Y_TRANSLATION
                ]:
                dof.constrained = True
                #print(f"Restringiendo nodo: {support_tag}, Rodillo en {dof.type_of_degree_of_freedom}, restringido?: {dof.constrained}, Tag-number: {dof.tag_number}")
    
    def _enumerate_degrees_of_freedom(self) -> None:
        """Assign consecutive DOF tag numbers in a deterministic priority order.

        Priority per joint (and globally):
          1. X_TRANSLATION
          2. Y_TRANSLATION
          3. Z_ROTATION

        First all *unconstrained* DOFs are numbered, then all *constrained* DOFs.
        """

        ordered_dof_types = [
            models.EnumTypesOfDegreeOfFreedom.X_TRANSLATION,
            models.EnumTypesOfDegreeOfFreedom.Y_TRANSLATION,
            models.EnumTypesOfDegreeOfFreedom.Z_ROTATION,
        ]

        def _enumerate(joint_index: int, dof_type_index: int, tag_number: int, constrained: bool) -> int:
            if joint_index >= len(self.structure.joints):
                return tag_number

            if dof_type_index >= len(ordered_dof_types):
                return _enumerate(joint_index + 1, 0, tag_number, constrained)

            joint = self.structure.joints[joint_index]
            desired_type = ordered_dof_types[dof_type_index]
            dof = next(
                (d for d in joint.degrees_of_freedom if d.type_of_degree_of_freedom == desired_type),
                None,
            )

            if dof and dof.constrained == constrained:
                #print(f"Enumerando grado libertad en Nodo: {joint.tag}")
                #print(f"Tipo de dof: {dof.type_of_degree_of_freedom}, Tag-number ANTES: {dof.tag_number}")
                dof.tag_number = tag_number
                tag_number += 1
                #print(f"Tag-number DESPUES: {dof.tag_number}")

            return _enumerate(joint_index, dof_type_index + 1, tag_number, constrained)

        tag_number = 1
        tag_number = _enumerate(0, 0, tag_number, constrained=False)
        _enumerate(0, 0, tag_number, constrained=True)

    def _count_degrees_of_freedom(self):
        self._count_dof = sum(len(joint.degrees_of_freedom) for joint in self.structure.joints)
        self._count_dof_constrained = sum(dof.constrained for joint in self.structure.joints for dof in joint.degrees_of_freedom)
        self._count_dof_unconstrained = self._count_dof - self._count_dof_constrained
    
    def _calculate_global_stiffness_matrix_from_structure(self) -> None:
        """Calculate the global stiffness matrix of the structure by assembling the stiffness matrices of each element according to their connectivity and DOF enumeration."""
        print(f"Cantidad total de grados de libertad en la estructura: {self._count_dof}")
        matrix = np.zeros((self._count_dof, self._count_dof))
        
        for element in self.structure.elements:
            print(f"\nMatrix of element: {element.tag}, length: {element.length}, angle: {element.inclination_angle}")
            #element.stiffness_matrix.print_matrix()
            #element.stiffness_matrix.print_symbolic_matrix()
            
            headers= []
            headers.extend([dof.tag_number for dof in element.initial_joint.degrees_of_freedom])
            headers.extend([dof.tag_number for dof in element.final_joint.degrees_of_freedom])
            print(f"Headers: {headers}")
            df = pd.DataFrame(element.stiffness_matrix.global_stiffness_matrix.tolist(), columns=headers, index=headers)
            print(f"DataFrame matriz del elemento:\n{df}")
            for i, fila in enumerate(headers):
                for j, columna in enumerate(headers):
                    matrix[fila-1, columna-1] += element.stiffness_matrix.global_stiffness_matrix[i, j]
        print("\nMatriz global de rigidez de la estructura:")
        dft = pd.DataFrame(matrix, columns=range(1, self._count_dof+1), index=range(1, self._count_dof+1))
        print(f"DataFrame matriz global:\n{dft}")
        #sp.pprint(sp.Matrix(matrix))
        self.global_stiffness_matrix = matrix

    def _calculate_global_equivalent_load_vector_from_structure(self) -> None:
        load_vector = np.zeros(self._count_dof)

        for equivalent_forces in self.structure.loads:
            vector = equivalent_forces.get_global_forces()
            
            headers= []
            headers.extend([dof.tag_number for dof in equivalent_forces.element.initial_joint.degrees_of_freedom])
            headers.extend([dof.tag_number for dof in equivalent_forces.element.final_joint.degrees_of_freedom])
            print(f"Headers: {headers}")
            for i, fila in enumerate(headers):
                load_vector[fila-1] += vector[i]
                print(f"fila: {fila}, i: {i}, valor: {vector[i]}, acumulado: {load_vector[fila-1]}")
        print("\nVector de cargas equivalentes [Feq_n] y [Feq_a]:")
        serie = pd.Series(load_vector, index=range(1, self._count_dof+1), name="Cargas equivalentes")
        print(serie)
        self.global_equivalent_load_vector = load_vector
        
    def _calculate_global_nodal_load_vector_from_structure(self) -> None:
        load_vector = np.zeros(self._count_dof)        
        #! TODO
        self.global_nodal_load_vector = load_vector
    
    def _calculate_global_known_displacements(self) -> None:
        displacements_vector = np.zeros(self._count_dof) 
        #[da]
        #! TODO
        self.global_displacements_vector = displacements_vector
    
    def _calculate_global_unknown_displacements(self) -> None:
        #[dn]
        Knn = self.global_stiffness_matrix[:self._count_dof_unconstrained, :self._count_dof_unconstrained]
        Kna = self.global_stiffness_matrix[:self._count_dof_unconstrained, self._count_dof_unconstrained:]
        print(f"\nGrados de libertad, Total: {self._count_dof}, Restringidos: {self._count_dof_constrained}, No Restringidos: {self._count_dof_unconstrained}")
        df_1 = pd.DataFrame(Knn, index=range(1, self._count_dof_unconstrained + 1), columns=range(1, self._count_dof_unconstrained + 1))
        df_2 = pd.DataFrame(Kna, index=range(1, self._count_dof_unconstrained + 1), columns=range(self._count_dof_unconstrained + 1, self._count_dof + 1))
        print(f"Submatriz [Knn]:\n{df_1}")
        print(f"Submatriz [Kna]:\n{df_2}")
        # Formula general [dn] = inv(Knn) * ([Kna] * [da] + [Fn] - [Feq_n])
        displacements_vector = np.linalg.inv(Knn) @ ((Kna @ self.global_displacements_vector[self._count_dof_unconstrained:]) + self.global_nodal_load_vector[:self._count_dof_unconstrained] - self.global_equivalent_load_vector[:self._count_dof_unconstrained])
        self.global_unknown_displacements_vector = displacements_vector
        serie = pd.Series(displacements_vector, index=range(1, self._count_dof_unconstrained+1), name="Desplazamientos desconocidos")#.to_frame(name="Desplazamientos desconocidos").style.format("{:.6e}")
        print(f"\n{serie}")
    
    def _calculate_global_unknown_reactions(self) -> None:
        Kan = self.global_stiffness_matrix[self._count_dof_unconstrained:, :self._count_dof_unconstrained]
        Kaa = self.global_stiffness_matrix[self._count_dof_unconstrained:, self._count_dof_unconstrained:]
        df_1 = pd.DataFrame(Kan, index=range(self._count_dof_unconstrained+1, self._count_dof + 1), columns=range(1, self._count_dof_unconstrained + 1))
        df_2 = pd.DataFrame(Kaa, index=range(self._count_dof_unconstrained+1, self._count_dof + 1), columns=range(self._count_dof_unconstrained + 1, self._count_dof + 1))
        print(f"Submatriz [Kan]:\n{df_1}")
        print(f"Submatriz [Kaa]:\n{df_2}")
        # Formula general [Fa] = [Kan] * [dn] + [Kaa] * [da] + [Feq_a] 
        reactions_vector = (Kan @ self.global_unknown_displacements_vector + Kaa @ self.global_displacements_vector[self._count_dof_unconstrained:]) + self.global_equivalent_load_vector[self._count_dof_unconstrained:]
        self.global_reactions_vector = reactions_vector
        serie = pd.Series(reactions_vector, index=range(self._count_dof_unconstrained+1, self._count_dof+1), name="Reacciones")#.to_frame(name="Reacciones").style.format("{:.6e}")
        print(f"\n{serie}")
    
    def run_analysis(self):
        """Run the structural analysis by performing the following steps:
        1. Constrain DOFs based on supports.
        2. Enumerate DOFs.
        3. Calculate global stiffness matrix.
        4. Apply loads and solve for displacements and reactions.
        """
        for support in self.structure.supports:
            self._constrain_degrees_of_freedom_from_support(support.tag)
        
        self._enumerate_degrees_of_freedom()
        self._count_degrees_of_freedom()
        self._calculate_global_stiffness_matrix_from_structure()
        self._calculate_global_equivalent_load_vector_from_structure()
        self._calculate_global_nodal_load_vector_from_structure()
        self._calculate_global_known_displacements()
        self._calculate_global_unknown_displacements()
        self._calculate_global_unknown_reactions()
