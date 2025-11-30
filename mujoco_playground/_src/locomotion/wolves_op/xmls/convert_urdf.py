import os
from dm_control import mjcf

input_file = "humanoid_v49.urdf"
output_file = "humanoid_converted.xml"

# Konvertierung
print(f"Konvertiere {input_file} → {output_file} ...")
model = mjcf.from_file(input_file)

# Speichern
with open(output_file, "w") as f:
    f.write(model.to_xml_string())

print(f"Fertig! Gespeichert als {output_file}")
