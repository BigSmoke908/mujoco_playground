import mujoco
import sys

input_file = "humanoid_v49.urdf"
output_file = "humanoid_converted.xml"

print(f"Konvertiere {input_file} zu {output_file}...")

model = mujoco.MjModel.from_xml_path(input_file)  # URDF wird automatisch geparst!
xml_string = mujoco.mj_saveLastXML(output_file, model)

print(f"✔ Fertig! Gespeichert als {output_file}")
