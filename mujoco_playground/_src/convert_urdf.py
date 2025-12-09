import mujoco
import argparse
import sys

parser = argparse.ArgumentParser(description="Konvertiere eine URDF-Datei zu MJCF XML.")
parser.add_argument("input_file", type=str, help="Pfad zur URDF-Datei, die konvertiert werden soll.")
parser.add_argument("output_file", type=str, help="Pfad zur Ausgabedatei für die MJCF XML.")

args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

print(f"Konvertiere {input_file} zu {output_file}...")

# URDF laden und in MJCF XML konvertieren
model = mujoco.MjModel.from_xml_path(input_file)
xml_string = mujoco.mj_saveLastXML(output_file, model)

print(f"✔ Fertig! Gespeichert als {output_file}")
