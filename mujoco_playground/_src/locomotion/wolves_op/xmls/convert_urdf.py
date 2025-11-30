from mujoco.mjcf_from_urdf import convert

# Eingabe-URDF und Ausgabe-XML
input_file = "humanoid_v49.urdf"
output_file = "humanoid_v49.xml"

convert(input_file, output_file)
print(f"Konvertierung abgeschlossen: {output_file}")
