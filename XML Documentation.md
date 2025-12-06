# Automatisierte XML Erzeugung
Um aus einer URDF File eine passende XML zu machen, muss man diese hierzu erstmal konvertieren. 

- URDF, Konvertierungsskript und STL Dateien in einen Ordner tun 
- Skript ausführen (python converter.py {Pfad Input.urdf} {Pfad Output.xml}
- Neue XML wird gespeichert im Ordner

----------------------------------------------------------------------------------------------------------------------------------
Konverter: 

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
----------------------------------------------------------------------------------------------------------------------------------

# Anpassungen an der grundlegenden XML:

## Neue globale Simulationseinstellungen

Die aktuelle XML enthält einen neuen <option>-Block, der vorher nicht existierte:
- Solver-Iterationen und Timestep definiert
- Integrator auf Euler gesetzt
- Euler-Dämpfung deaktiviert
Zweck: Stabilere/angepasste Simulation für den Roboter.



## Einführung von Default-Klassen (Große strukturelle Erweiterung)

Neu hinzugefügt wurde ein kompletter <default>-Abschnitt mit:
- Standard-Kollisionsdefinitionen
- Standard-Visual-Definitionen
- Motor-/Servo-Klassen: mx106, mx64, xh-540
- Standard-Sites
Vereinheitlichte Basiseinstellungen für alle Körperteile & Motorparameter.


## Änderungen am Worldbody
- Die neue Datei hat ein <light> in worldbody
- Torso Position von pos="0 0 0.6" auf pos="0 0 0.535" geändert
- <site name="imu"/> jetzt direkt am Torso positioniert
Zweck: Physikalisch korrektere IMU-Platzierung und Beleuchtung.

## Meshes
Torso-Meshes haben ein zusätzliches Attribut: class="visual"


## Gelenke & Arm-Kinematik
Entfernen der Arme für einfachere Simulation


## Actuatoren
Die Aktuatorstruktur wurde komplett ersetzt:Im Original Motoren pro Arm, Bein, Kopf — jeweils <motor> mit gear & ctrlrange
Jetzt: Nur noch Beinmotoren, aber als <position>-Controller Mit motor Klassen z. B. class="mx106"
Für den Positionsregler in der Simulation relevant


## Sensoren
Sensoren waren nicht mit in der Origininalen XML. Die neue XML enthält folgende Sensoren (je nachdem was in der physischen Variante verbaut ist):
- Gyro
- IMU-Velocimeter
- Accelerometer
- Orientierung/Positionsframe
- Fußsensoren (links/rechts)
