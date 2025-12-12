# Mujoco Modell Dokumentation

> Hier wird die Generierung eines trainerfähigen Mujoco-Modells aus einer URDF-Datei beschrieben


## URDF zu Mujoco Model

Die Erstellung der Simulationsumgebung erfolgt in drei logischen Schritten. Es existiert ein prototypisches [Skript](../mujoco_playground/_src/convert_urdf.py) für den Prozess, dieses deckt aktuell aber nur Schritt 1 ab:

### Export (URDF zu XML): Konvertierung der URDF-Datei in eine rohe Mujoco-XML (`mjmodel.xml`)

 - [den Mujoco-Viewer öffnen](../README.md#mujoco-viewer-öffnen)
 - eine URDF-Datei reinziehen (wichtig: verwendete .stl-Files müssen sich neben der URDF befinden)
 - das Modell über "Save xml" abspeichern
 - an Stelle wo der python-Aufruf für den mujoco-viewer gestartet wurde befindet sich jetzt auch Basis-XML
 - diese Basis-XML muss jetzt noch in das Environment nach `xmls/wolvesop_mjx_footonly.xml` verlegt werden, wo diese in den nächsten Schritten weiter verarbeitet wird


### Aufbereitung des Modells:

> die URDF-Datei enthält nur einen Teil der für die Simulation benötigten Daten. Die fehlende Daten müssen manuell eingefügt werden. Die folgenden Schritte sind stark an dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) orientiert und daher teilweise nicht extra begründet. Das in unserem Projekt entwickelte Roboter-Modell befindet sich in der [wolvesop_mjx_footonly.xml](../mujoco_playground/_src/locomotion/wolves_op/xmls/wolvesop_mjx_feetonly.xml).

#### Simulations-Parameter

Direkt in dem `mujoco`-Element müssen folgende Simulations-Parameter eingefügt werden. Diese wurden von dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) entnommen:

```xml
<option iterations="3" ls_iterations="5" timestep="0.002" integrator="Euler">
    <flag eulerdamp="disable"/>
</option>
```

#### Mujoco-Basisklassen

Mujoco erlaubt die Definition von Basisklassen, welche anschließend für Default-Werte für andere Elemente vorgeben können. Die folgenden Basisklassen wurden aus dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) entnommen, um die Kollision von Bauteilen und Servo-Parameter zu setzen:

```xml
<default>
    <default class="humanoid">

        <default class="collision">
            <geom group="3" contype="1" conaffinity="1" />
        </default>
        
        <default class="visual">
            <geom contype="0" conaffinity="0" group="2" />
        </default>

        <!-- Motorklassen mit unterschiedlichen Stärken ! -->
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

Dieser Block befindet sich ebenfalls direkt in dem `mujoco`-Element.


#### Asset-Pfade

In dem eingerichteten Wolves-OP Environment sind die .stl-Dateien zur besseren Übersicht in `xmls/stls/` platziert. Damit diese richtige gefunden werden, müssen die Dateipfade aller meshes angepasst werden. Diese befinden sich in `mujoco>asset>mesh`. So sieht eine beispielhafte Anpassung aus:

Vorher:
```xml
<mesh name="base_link" content_type="model/stl" file="base_link.stl" scale="0.001 0.001 0.001"/>
```

Nachher:
```xml
<mesh name="base_link" content_type="model/stl" file="stls/base_link.stl" scale="0.001 0.001 0.001"/>
```

#### Kollisionen einstellen

Für das Training können bei allen Bauteilen außer den Fußplatten die Kollisionen deaktiviert werden. Hierfür wird bei allen `geom`-Element die vorher definierte Basisklasse `visual` verwendet. Ein Beispiel:

Vorher:
```xml
<geom type="mesh" rgba="0 0 0.54 1" mesh="base_link"/>
```

Nachher:
```xml
<geom class="visual" type="mesh" rgba="0 0 0.54 1" mesh="base_link" />
```

Eine Ausname dazu stellen die Fußplatten dar. Bei allen Bauteilen, welche Teile der Linken/Rechten Fußplatte sind, wird stattdessen die Klasse `collision` verwendet:

Vorher:
```xml
<geom pos="-0.069743 -0.100908 0.494126" quat="1 0 0 0" type="mesh" rgba="0.13 0.16 0.21 1" mesh="22_foot_plate_left_v2_1"/>
```

Nachher:
```xml
<geom class="collision" pos="-0.069743 -0.100908 0.494126" quat="1 0 0 0" type="mesh" rgba="0.13 0.16 0.21 1" mesh="22_foot_plate_left_v2_1" />
```


#### Freejoint und IMU einfügen

In das `worldbody`-Element müssen die IMU und ein Freejoint eingefügt werden. Die IMU kann alternativ an anderen Stellen (je nach Robotermodell) platziert werden:

```xml
<freejoint />
<site name="imu" pos="0.06 0.045 0.000"/>
```


#### Torso einfügen

Um die Struktur an die Struktur in dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) anzupassen muss der gesamte Inhalt von dem `worldbody`-Element in ein weiteres `body`-Element platziert werden:

Vorher:
```xml
<worldbody>
    ...
</worldbody>
```

Nachher:
```xml
<worldbody>
    <body name="torso" pos="0 0 0.535" childclass="humanoid">
        ...
    </body>
</worldbody>
```

Das `pos`-Attribut muss noch angepasst werden, falls der Roboter zu Beginn der Szene im Boden spawnt.


#### Fußplatten konfigurieren

Die `geom`-Element mit den Meshes der Fußplatten werden später innerhalb von dem Script angesprochen und benötigen daher eigene Namen (`l_foot1` bzw. `r_foot1`):

Vorher:
```xml
<geom class="collision" pos="-0.069743 -0.100908 0.494126" quat="1 0 0 0" type="mesh" rgba="0.13 0.16 0.21 1" mesh="22_foot_plate_left_v2_1" />
```

Nachher
```xml
<geom name="l_foot1" class="collision" pos="-0.069743 -0.100908 0.494126" quat="1 0 0 0" type="mesh" rgba="0.13 0.16 0.21 1" mesh="22_foot_plate_left_v2_1" />
```


Außerdem müssen neben den Meshes für die Fußplatten je eine `site`-platziert werden, damit hier später simulierte Sensoren angebracht werden können:

```xml
<site name="l_foot" />
```


#### Actuator-Klassen anwenden

Die vorher definierten Actuator-Klassen müssen für das gesamte Modell verwendet werden. Das kann erreicht werden, in dem jeweils das Startelement der verschiedenen Extremitäten diese als `childclass`-erhalten, sodass die Actuator-Klasse für alle Joints angewendet wird:

Vorher
```xml
<body name="HN05-N101_v1__18__1" pos="0.059842 -0.008192 -0.1063">
```

Nachher
```xml
<body childclass="mx106" name="HN05-N101_v1__18__1" pos="0.059842 -0.008192 -0.1063">
```

(Dieser Teil kann nur teilweise auf einen anderen Roboter angewendet werden, da davon ausgegangen wird, dass alle Joints in einer Extremität den gleichen Servo-Typen verwenden)


#### Joint Konfiguration

Der Einfachheit halber wurden die Joint-Namen aus dem Basisprojekt in die XML übernommen. Daher müssen alle Joints in den Beinen wie folgt umbenannt werden:

| Original Name (mjmodel.xml) | Status         | Neuer Name (wolvesop_mjx_feetonly.xml) | Beschreibung           |
| :-------------------------- | :------------- | :------------------------------------- | :--------------------- |
| **Rechtes Bein**            |                |                                        |                        |
| hip_yaw_r                   | Umbenannt      | LR_HR                                  | Hüfte Rotation (Yaw)   |
| hip_roll_r                  | Umbenannt      | LR_HAA                                 | Hüfte Abduktion (Roll) |
| hip_pitch_r                 | Umbenannt      | LR_HFE                                 | Hüfte Flexion (Pitch)  |
| knee_r                      | Umbenannt      | LR_KFE                                 | Knie Flexion (Pitch)   |
| ankle_pitch_r               | Umbenannt      | LR_FFE                                 | Fuß Flexion (Pitch)    |
| ankle_roll_r                | Umbenannt      | LR_FAA                                 | Fuß Abduktion (Roll)   |
| **Linkes Bein**             |                |                                        |                        |
| hip_yaw_l                   | Umbenannt      | LL_HR                                  | Hüfte Rotation (Yaw)   |
| hip_roll_l                  | Umbenannt      | LL_HAA                                 | Hüfte Abduktion (Roll) |
| hip_pitch_l                 | Umbenannt      | LL_HFE                                 | Hüfte Flexion (Pitch)  |
| knee_l                      | Umbenannt      | LL_KFE                                 | Knie Flexion (Pitch)   |
| ankle_pitch_l               | Umbenannt      | LL_FFE                                 | Fuß Flexion (Pitch)    |
| ankle_roll_l                | Umbenannt      | LL_FAA                                 | Fuß Abduktion (Roll)   |


Das erfolgt in den jeweils dafür vorgesehenen `joint`-Elementen:

Vorher
```xml
<joint name="hip_yaw_l" pos="0 0 0" axis="0 0 1" range="-3.14159 3.14159" actuatorfrcrange="-8.4 8.4"/>
```

Nachher
```xml
<joint name="LL_HR" pos="0 0 0" axis="0 0 1" range="-3.14159 3.14159" actuatorfrcrange="-8.4 8.4" />
```


Die Joints in den Armen und dem Kopf werden von der trainierten Policy nicht verwendet, sodass die folgenden Joints einfach auskommentiert oder entfernt werden können:

| Original Name (mjmodel.xml) | Status         | Neuer Name (wolvesop_mjx_feetonly.xml) | Beschreibung           |
| :-------------------------- | :------------- | :------------------------------------- | :--------------------- |
| **Kopf & Hals**             |                |                                        |                        |
| head_pan                    | Auskommentiert | -                                      | Kopf Drehung (Hals)    |
| head_tilt                   | Auskommentiert | -                                      | Kopf Neigung           |
| **Linker Arm**              |                |                                        |                        |
| shoulder_pitch_l            | Auskommentiert | -                                      | Schulter Pitch         |
| shoulder_roll_l             | Auskommentiert | -                                      | Schulter Roll          |
| elbow_l                     | Auskommentiert | -                                      | Ellbogen               |
| **Rechter Arm**             |                |                                        |                        |
| shoulder_pitch_r            | Auskommentiert | -                                      | Schulter Pitch         |
| shoulder_roll_r             | Auskommentiert | -                                      | Schulter Roll          |
| elbow_r                     | Auskommentiert | -                                      | Ellbogen               |


#### Actuator Konfiguration

An den Joints welche von der Policy angesteuert werden werden jetzt Aktuatoren mit den vorher definierten Motorklassen eingefügt. Das passiert wieder innerhalb von dem `mujoco`-Element. Diese Elemente müssen eingefügt werden:

```xml
<actuator>
    <!-- Right leg (Positionsregler) -->
    <position joint="LR_HR" name="LR_HR" class="mx106" />
    <position joint="LR_HAA" name="LR_HAA" class="mx106" />
    <position joint="LR_HFE" name="LR_HFE" class="mx106" />
    <position joint="LR_KFE" name="LR_KFE" class="mx106" />
    <position joint="LR_FFE" name="LR_FFE" class="mx106" />
    <position joint="LR_FAA" name="LR_FAA" class="mx106" />

    <!-- Left leg (Positionsregler) -->
    <position joint="LL_HR" name="LL_HR" class="mx106" />
    <position joint="LL_HAA" name="LL_HAA" class="mx106" />
    <position joint="LL_HFE" name="LL_HFE" class="mx106" />
    <position joint="LL_KFE" name="LL_KFE" class="mx106" />
    <position joint="LL_FFE" name="LL_FFE" class="mx106" />
    <position joint="LL_FAA" name="LL_FAA" class="mx106" />
</actuator>
```

#### Sensor-Konfiguration

Ebenfalls in dem `mujoco`-Element werden abschließend die simulierten Sensoren platziert. Diese dienen (mit Ausnahme vom `gyro`, dem `velocimeter` und dem `accelerometer`) ausschließlich der Berechnung für die Reward-Function, sodass auch Sensorik die nicht real existiert verwendet wird:

```xml
<sensor>
    <gyro site="imu" name="gyro"/>
    <velocimeter site="imu" name="local_linvel"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framexaxis objtype="site" objname="imu" name="forwardvector"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <framepos objtype="site" objname="imu" name="position"/>
    <framequat objtype="site" objname="imu" name="orientation"/>

    <framelinvel objtype="site" objname="l_foot" name="l_foot_global_linvel"/>
    <framelinvel objtype="site" objname="r_foot" name="r_foot_global_linvel"/>
    <framexaxis objtype="site" objname="l_foot" name="l_foot_upvector"/>
    <framexaxis objtype="site" objname="r_foot" name="r_foot_upvector"/>
    <framepos objtype="site" objname="l_foot" name="l_foot_pos"/>
    <framepos objtype="site" objname="r_foot" name="r_foot_pos"/>
</sensor>
```


Die Konfiguration Aufbereitung des eigentlichen Roboter Modells ist damit abgeschlossen und dieses kann in den nächsten Schritten in eine Szene eingefügt werden.


### Komposition (Scene XML): Einbettung des Roboter-MOdells in eine Umgebung mit Boden

Bei dem Training wird nicht das eigentliche Roboter-Modell (`wolvesop_mjx_feetonly.xml`) geladen, sondern eine Szene verwendet welche dieses Modell beinhaltet. Für das normale Training verwenden wir hier die [`scene_mjx_feetonly_flat_terrain.xml](../mujoco_playground/_src/locomotion/wolves_op/xmls/scene_mjx_feetonly_flat_terrain.xml). Diese ist bis auf wenige Änderungen vollständig aus dem Basis-Environment übernommen # TODO hier auf Environment-Doku verlinken.

Die einzigen vorgenommenen Änderungen sind hier die inkludierte Datei `<include file="wolvesop_mjx_feetonly.xml" />` und der zu Beginn des Trainings geladene `key`, über welchen die Startpositionen der Servos eingestellt werden:

```xml
<keyframe>
    <key name="home" qpos="
        0 0 0.56
        1 0 0 0
        0.023628265148262724 -0.11 -0.3 1.1 0.5495038397740458 -0.12913515511895796 
        -0.016441795868928723 0.11 0.3 -1.1 -0.5537397918567754 0.07437380704149316" ctrl="
        0 0 0 0 0 0
        0 0 0 0 0 0" />
</keyframe>
```


## Dateistruktur & Assets

Ähnlich wie eine `URDF`-Datei verwenden auch Mujoco-XML-Modell im Hintergrund .stdl-Dateien um Meshes zu laden. Diese müssen in `xmls/stls/` platziert werden.

