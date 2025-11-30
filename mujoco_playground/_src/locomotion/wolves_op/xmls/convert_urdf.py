import os
from dm_control.mujoco.wrapper import mjcf
from dm_control.suite.utils import parse_urdf

# Eingabe / Ausgabe
input_file = "humanoid_v49.urdf"
output_file = "humanoid_converted.xml"

print(f"Konvertiere {input_file} → {output_file} ...")

# URDF einlesen und umwandeln
model = parse_urdf(input_file)

# In MuJoCo XML umwandeln
mjcf_root = mjcf.from_mjcf_model(model)
xml_data = mjcf_root.to_xml_string()

# XML speichern
with open(output_file, "wb") as f:  # ACHTUNG → "wb" wegen Bytes
    f.write(xml_data)

print(f"✔ XML erfolgreich gespeichert unter: {output_file}")
