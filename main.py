from fastapi import FastAPI
from src.models import models
from src.services import services
import math

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from StructureSolver!"}

def main():
    print("Hello from StructureSolver!")
    print("\n#############################")   
    print("##### Ejemplo de uso-1 ######")
    print("#############################")   
    cross_section = models.CrossSection(A=0.06, I=0.00045)
    structure_service = services.StructureSolverService()
    structure_service.add_joint_to_structure(x=0.0, y=4.0, name="A")
    structure_service.add_joint_to_structure(x=4.0, y=4.0, name="B")
    structure_service.add_joint_to_structure(x=5.0, y=0.0, name="C")
    structure_service.add_element_to_structure("A", "B", cross_section, 2213594.362, "A-B")
    structure_service.add_element_to_structure("B", "C", cross_section, 2213594.362, "B-C")
    structure_service.add_support_to_structure("A", models.EnumSupportType.FIXED, name="SupportA")
    structure_service.add_support_to_structure("C", models.EnumSupportType.FIXED, name="SupportC")
    structure_service.add_puntual_load_to_element("A-B", 5.0, 2.0)
    structure_service.add_distributed_load_to_element("B-C", 4.0, 4.0, 0.0, 4.123105625617661)
    structure_service.run_analysis()
    
    
    print("\n#############################")   
    print("##### Ejemplo de uso-2 ######")
    print("#############################")
    seccion_vigas = models.CrossSection(A=0.25*0.50, I=0.25*0.50**3/12)
    seccion_columnas = models.CrossSection(A=0.25*0.25, I=0.25*0.25**3/12)
    #young_modulus = 14000*math.sqrt(210)
    young_modulus = 15100*math.sqrt(210) # f'c= 210 kg/cm2
    young_modulus = young_modulus*(1/0.0001) # kg/m2
    structure_service_2 = services.StructureSolverService()
    # Nodos
    structure_service_2.add_joint_to_structure(x=0.0, y=0.0, name="A")
    structure_service_2.add_joint_to_structure(x=5.0, y=0.0, name="B")
    structure_service_2.add_joint_to_structure(x=9.0, y=0.0, name="C")
    structure_service_2.add_joint_to_structure(x=0.0, y=4.0, name="D")
    structure_service_2.add_joint_to_structure(x=5.0, y=4.0, name="E")
    structure_service_2.add_joint_to_structure(x=9.0, y=4.0, name="F")
    structure_service_2.add_joint_to_structure(x=10.5, y=4.0, name="G")
    structure_service_2.add_joint_to_structure(x=0.0, y=7.0, name="H")
    structure_service_2.add_joint_to_structure(x=5.0, y=7.0, name="I")
    structure_service_2.add_joint_to_structure(x=9.0, y=7.0, name="J")
    structure_service_2.add_joint_to_structure(x=10.5, y=7.0, name="K")
    # Elementos
    structure_service_2.add_element_to_structure("A", "D", seccion_columnas, young_modulus, "Col-A-D")
    structure_service_2.add_element_to_structure("D", "H", seccion_columnas, young_modulus, "Col-D-H")
    structure_service_2.add_element_to_structure("B", "E", seccion_columnas, young_modulus, "Col-B-E")
    structure_service_2.add_element_to_structure("E", "I", seccion_columnas, young_modulus, "Col-E-I")
    structure_service_2.add_element_to_structure("C", "F", seccion_columnas, young_modulus, "Col-C-F")
    structure_service_2.add_element_to_structure("F", "J", seccion_columnas, young_modulus, "Col-F-J")
    structure_service_2.add_element_to_structure("D", "E", seccion_vigas, young_modulus, "Vig-D-E")
    structure_service_2.add_element_to_structure("E", "F", seccion_vigas, young_modulus, "Vig-E-F")
    structure_service_2.add_element_to_structure("F", "G", seccion_vigas, young_modulus, "Vig-F-G")
    structure_service_2.add_element_to_structure("H", "I", seccion_vigas, young_modulus, "Vig-H-I")
    structure_service_2.add_element_to_structure("I", "J", seccion_vigas, young_modulus, "Vig-I-J")
    structure_service_2.add_element_to_structure("J", "K", seccion_vigas, young_modulus, "Vig-J-K")
    # Apoyos
    structure_service_2.add_support_to_structure("A", models.EnumSupportType.FIXED, name="Apoyo-A")
    structure_service_2.add_support_to_structure("B", models.EnumSupportType.FIXED, name="Apoyo-B")
    structure_service_2.add_support_to_structure("C", models.EnumSupportType.FIXED, name="Apoyo-C")
    # Cargas
    structure_service_2.add_distributed_load_to_element("Vig-D-E", 700, 700, 0.0, 5)
    structure_service_2.add_distributed_load_to_element("Vig-E-F", 700, 700, 0.0, 4)
    structure_service_2.add_distributed_load_to_element("Vig-F-G", 700, 700, 0.0, 1.5)
    structure_service_2.add_distributed_load_to_element("Vig-H-I", 700, 700, 0.0, 5)
    structure_service_2.add_distributed_load_to_element("Vig-I-J", 700, 700, 0.0, 4)
    structure_service_2.add_distributed_load_to_element("Vig-J-K", 700, 700, 0.0, 1.5)
    
    structure_service_2.run_analysis()

if __name__ == "__main__":
    main()
