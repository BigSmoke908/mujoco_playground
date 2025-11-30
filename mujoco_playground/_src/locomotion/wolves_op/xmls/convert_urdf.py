import os
from dm_control.mjcf.urdf import parser

input_file = "humanoid_v49.urdf"
output_file = "humanoid_converted.xml"

print(f"Konvertiere {input_file} → {output_file} ...")

# URDF -> MJCF Konvertierung
converter = parser.UrdfToMjcf()
model = converter.parse(input_file)

# Als XML speichern
with open(output_file, "w") as f:
    f.write(model.to_xml_string())

print(f"Fertig! Gespeichert als {output_file}")
