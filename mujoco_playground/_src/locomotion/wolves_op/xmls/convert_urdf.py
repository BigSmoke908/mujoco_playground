from dm_control import mjcf
from dm_control.utils import xml_util

# Eingabe und Ausgabe
input_file = "humanoid_v49.urdf"
output_file = "humanoid_v49.xml"

# Konvertieren
mj_model = mjcf.from_path(input_file)  # lädt URDF
xml_str = mj_model.to_xml_string()     # als MJCF XML

# Speichern
with open(output_file, "wb") as f:
    f.write(xml_str)

print(f"Konvertierung abgeschlossen: {output_file}")
