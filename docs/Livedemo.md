# Dokumentation zur Livedemo

## Quick Guide

Dieser Abschnitt beschreibt die minimal notwendigen Schritte, um die Livedemo **direkt aus dem Projekt-Root** zu starten.

### Voraussetzungen

* vollständig eingerichtete [Mujoco-Playground-Umgebung](../README.md#entwicklungsumgebung-einrichten)
* vorhandene ONNX-Datei im vorgesehenen Ordner (wichtig: das für das Training verwendete Environment muss mit dem aktuellen Stand übereinstimmen)

### Schritte

1. Environment aktivieren:

```bash
source .venv/bin/activate
```

2. Livedemo starten:

```bash
python mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py
```

3. Steuerung:

* W / A / S / D – Bewegung
* Q / E – Rotation
* Space – Reset
* die Konsole muss fokussiert sein

Nach dem Start öffnet sich der MuJoCo-Viewer und das Modell kann interaktiv gesteuert werden.

---

## 1. Ziel der Livedemo

Ziel der Livedemo ist es, das trainierte Modell interaktiv vorzuführen. Die Demo basiert auf der [Livedemo im Basisprojekt](../mujoco_playground/experimental/sim2sim/play_wolfgang_joystick.py) und wurde so angepasst, dass sie mit dem aktuellen Modell kompatibel ist. Die Steuerung erfolgt während der Laufzeit über die Tastatur (W, A, S, D, Q, E).

---

## 2. Ausgangsbasis

Als Grundlage diente folgende bestehende Livedemo:

> Quelle: [play_wolfgang_joystick.py](../mujoco_playground/experimental/sim2sim/play_wolfgang_joystick.py)

Die ursprüngliche Demo enthält bereits:

* Konsolenanwendung
* Einbindung eines ONNX-Modells
* Einfache Tastatursteuerung

Die auf die wolvesOP-Plattform angepasst Livedemo befindet sich [hier](../mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py).

---

## 3. Vorbereitung: ONNX-Modell

### 3.1 Konvertierung nach ONNX

Das trainierte Modell muss zunächst in das ONNX-Format konvertiert werden. Eine Anleitung befindet sich hier:

* [create_onnx.md](docs/create_onnx.md)

Das Ergebnis ist eine ONNX-Datei:

[wolves_op_policy.onnx](mujoco_playground/experimental/sim2sim/onnx/wolves_op_policy.onnx)

### 3.2 Ablage der ONNX-Datei

Die ONNX-Datei muss im folgenden Verzeichnis platziert werden:

```
mujoco_playground/
└── experimental/
    └── sim2sim/
        └── onnx/
            └── wolves_op_policy.onnx
```

---

## 4. Aufbau der Livedemo

### 4.1 Laden der ONNX-Datei

Der Speicherort der ONNX-Datei wird relativ zur Demo-Datei bestimmt:

```python
_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
```

Der Dateiname kann im Code angepasst werden:

```python
policy_path=(_ONNX_DIR / "wolves_op_policy.onnx").as_posix(),
```

---

### 4.2 Observations (Modelleingang)

Die Methode `get_obs(...)` erzeugt den Eingabevektor für das ONNX-Modell:

```python
obs = np.hstack([
    gyro,
    gravity,
    command,
    joint_angles,
    joint_velocities,
    self._last_action,
    phase,
])
```

Der Aufbau muss dem Beobachtungsraum des trainierten Modells entsprechen.

---

### 4.3 Actions (Modellausgang)

Die Kommunikation erfolgt in `get_control(...)`:

```python
onnx_input = {"obs": obs.reshape(1, -1)}
onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
```

Die Ausgabe wird als Steuerkommando verwendet:

```python
data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
```

---


### 4.4 MuJoCo-Initialisierung

Beim Start werden:

* das MuJoCo-Modell (XML + Assets)
* die Simulation
* die ONNX-Policy

initialisiert. Der Controller wird als Callback registriert:

```python
mujoco.set_mjcb_control(policy.get_control)
```

---

## 5. Start der Livedemo

Die Simulation wird über folgende Datei gestartet:

```
python experimental/sim2sim/play_wolvesOP_joystick.py
```

Nach dem Start öffnet sich der MuJoCo-Viewer und das Modell kann interaktiv gesteuert werden.

---

## 6. Anpassung für eigene Modelle

Für ein eigenes Modell müssen in der Startdatei folgende Punkte angepasst werden:

 - Erzeugung des Observationspace in `get_obs(...)`
 - Interpretation der Actions in `get_control(...)`
 - ONNX-Pfad
```python
policy_path=(_ONNX_DIR / "wolves_op_policy.onnx").as_posix(),
```
 - verlinktes Mujoco-Environment ändern
```python
model = mujoco.MjModel.from_xml_path(
    wolvesop_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    assets=get_assets(),
)
```

---

## 7. Zusammenfassung

Die Livedemo ermöglicht eine interaktive Vorführung eines trainierten ONNX-Modells in MuJoCo. Durch klare Ordnerstrukturen, eine einfache Tastatursteuerung und minimale Anpassungen lässt sich die Demo leicht reproduzieren und auf andere Modelle übertragen.
