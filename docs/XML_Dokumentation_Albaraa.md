# XML Dokumentation & Modell-Pipeline

Diese Dokumentation beschreibt die Struktur der XML-Modelle für den **WolvesOP** Roboter, den Workflow von der URDF zur MJX-optimierten Simulation und die Organisation der Assets.

## 1. Workflow & Pipeline

Die Erstellung der Simulationsumgebung erfolgt in drei logischen Schritten:

1.  **Export (URDF → XML):** Konvertierung der CAD/URDF-Daten in eine rohe Mujoco-XML (`mjmodel.xml`). Dies geschieht oft über Skripte (z.B. `converter.py`) oder manuelles hineinziehen der URDF-Datei in der Mujoco-Viewer und dann auf 
2.  **Optimierung (XML → MJX XML):** Manuelle Bereinigung und Erweiterung der Rohdaten. Hier entstehen Dateien wie `wolvesop_mjx_feetonly.xml`. Wichtige Schritte sind das Hinzufügen von Physik-Parametern, Sensoren und RL-gerechten Aktuatoren.
3.  **Komposition (Scene XML):** Einbettung des Roboters in eine Umgebung mit Licht und Boden (z.B. `scene_mjx_feetonly_flat_terrain.xml`).

<details>
<summary>Hier das converter.py Skript</summary>

```python
import mujoco
import argparse

# Nutzung: python converter.py input.urdf output.xml
parser = argparse.ArgumentParser(description="Konvertiere URDF zu MJCF XML.")
parser.add_argument("input_file", type=str, help="Pfad zur URDF-Datei")
parser.add_argument("output_file", type=str, help="Pfad zur Ausgabedatei")
args = parser.parse_args()

print(f"Konvertiere {args.input_file} zu {args.output_file}...")
model = mujoco.MjModel.from_xml_path(args.input_file)
mujoco.mj_saveLastXML(args.output_file, model)
print("✔ Fertig!")
```
</details>

---

## 2. Dateistruktur & Assets

Damit Mujoco die Dateien korrekt findet, ist die Ordnerstruktur essenziell. Alle relativen Pfade in den XMLs beziehen sich auf diese Hierarchie.

```text
mujoco_playground/_src/locomotion/wolves_op/xmls/
├── wolvesop_mjx_feetonly.xml          # Das optimierte Robotermodell (Hauptdatei)
├── scene_mjx_feetonly_flat_terrain.xml # Die Szene (lädt den Roboter)
├── scene_mjx_feetonly_rough_terrain.xml # Die Szene (lädt den Roboter)
├── mjmodel.xml                        # Der rohe Export (Referenz)
└── stls/                              # Ordner für Mesh-Dateien
    ├── humanoid_v49.urdf                  # Die Quelle (URDF)
    ├── base_link.stl
    ├── 25_Orin_Baseplate_v14_1.stl
    ├── ... (weitere Bauteile)
```

> **Hinweis:** In der optimierten XML nutzen wir `<compiler meshdir="stls"/>`. Dadurch können wir in den `<geom>`-Tags einfach `file="bauteil.stl"` schreiben, statt `file="stls/bauteil.stl"`.

-----

## 3\. Vergleich: Roh-Export vs. Optimiertes Modell

Hier ist der Unterschied zwischen der frisch exportierten `mjmodel.xml` und unserer angepassten `wolvesop_mjx_feetonly.xml`.

### Analyse der `mjmodel.xml` (Der Input)

Die `mjmodel.xml` ist sehr flach strukturiert. Sie definiert viele Körper mit generierten Namen wie `<body name="MX64_body_modified_v2__2__1">`. Sie ist visuell korrekt, aber für Reinforcement Learning oft zu komplex und instabil.

### Analyse der `wolvesop_mjx_feetonly.xml` (Das Ziel)

Diese Datei wurde massiv angepasst, um mit MJX (GPU-Simulation) zu funktionieren:

  * **Stabilität:** Der Integrator wurde auf `euler` gesetzt, da Runge-Kutta in differenzierbaren Simulationen oft zu Instabilität führt.
  * **Effizients:** Wir haben alle Teile des Roboters der Klasse visuell `<geom class="visual"` zugeordnet außer die Fußplatten `<geom name="r_foot1" class="collision"` & `<geom name="l_foot1" class="collision"`. Für das Training muss somit keine rechenintensive Kollisionen berechnet werden, welche irrelevant für das Training sind.
  * **Wartbarkeit:** Durch die Nutzung von `<default class="...">` Blöcken können wir die Steifigkeit (Stiffness/Damping) aller Knie-Motoren gleichzeitig ändern, ohne 12 einzelne Gelenke editieren zu müssen.
  * **Beobachtungen:** Ein `<sensor>`-Block liefert dem Agenten die notwendigen Daten (Orientation, Angular Velocity).

-----

## 4\. Details der XML-Komponenten

### A. Globale Einstellungen (`<option>`)

Wir nutzen spezifische Einstellungen für die Stabilität in MJX, welche von Wolfgang übernommen worden sind:

```xml
<option timestep="0.002" iterations="1" integrator="euler" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
</option>
```

* `iterations="1"`: Typisch für MJX/Brax, da auf der GPU viele kleine Schritte effizienter sind.

### B. Defaults & Klassen
Wir nutzen eine hierarchische Struktur für die Defaults. Die Hauptklasse humanoid enthält Unterklassen für Kollisionen, Visualisierung und die verschiedenen Motortypen (Dynamixel MX-106, MX-64, XH-540).

Dies erlaubt eine sehr saubere Definition der Körper (body), da man nur noch die Klasse angeben muss, statt Dämpfung und Reibung jedes Mal neu zu definieren.

```xml
<default>
    <default class="humanoid">
        <default class="collision">
            <geom group="3" contype="1" conaffinity="1" />
        </default>
        <default class="visual">
            <geom contype="0" conaffinity="0" group="2" />
        </default>

        <default class="mx106">
            <joint damping="1.7" armature="0.025" frictionloss="0.1" />
            <position kp="21.0" ctrlrange="-3.141592 3.141592" forcerange="-8.4 8.4" />
        </default>
        <default class="mx64">
            <joint damping="0.66" armature="0.012" frictionloss="0.09" />
            <position kp="12.5" ctrlrange="-3.141592 3.141592" forcerange="-6 6" />
        </default>
        <default class="xh-540">
            <joint damping="1.7" armature="0.025" frictionloss="0.1" />
            <position kp="21.0" ctrlrange="-3.141592 3.141592" forcerange="-8.4 8.4" />
        </default>
        
        <site size="0.01" rgba="1 0 0 1" group="4"/>
    </default>
</default>
```

### C. Worldbody & Sites

  * **Torso-Position:** Wurde angepasst (z.B. `pos="0 0 0.535"`), damit der Roboter korrekt auf dem Boden steht.
  * **IMU-Site:** Eine `<site name="imu" .../>` wurde im Rumpf platziert. Diese ist unsichtbar, dient aber als Ankerpunkt für die IMU-Sensoren.

### D. Aktuatoren (Motoren)

Wir verwenden Positions-Regelung (Position Control), da dies für Dynamixel-Servos am realistischsten ist:

```xml
<actuator>
    <position name="r_hip_pitch" joint="r_hip_pitch_joint" class="mx106" user="1"/>
    </actuator>
```

-----

## 5\. Szenen-Komposition

Die Datei `scene_mjx_feetonly_flat_terrain.xml` ist der Einstiegspunkt für das Training. Sie definiert **nicht** den Roboter neu, sondern inkludiert ihn.

**Struktur der Szene:**

1.  **`<include file="wolvesop_mjx_feetonly.xml"/>`**: Lädt den Roboter.
2.  **Visuals:** Definiert `<statistic>`, `<visual>` (Headlight, RGBA) und `<skybox>`.
3.  **Umgebung:** Definiert den Boden (`<geom name="floor" type="plane"...>`).


```xml
<mujoco model="WolvesOP Scene">
  <include file="wolvesop_mjx_feetonly.xml"/>
  
  <statistic center="0 0 0.5" extent="0.8"/>
  
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
  </worldbody>
</mujoco>
```

Durch diese Trennung können wir den gleichen Roboter einfach in verschiedene Umgebungen (Flat Terrain, Rough Terrain) setzen, ohne die Roboter-XML kopieren zu müssen.