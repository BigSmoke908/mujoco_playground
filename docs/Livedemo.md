# Dokumentation zur Livedemo

## Quick Guide (Kurzstart)

Dieser Abschnitt beschreibt die minimal notwendigen Schritte, um die Livedemo **direkt aus dem Projekt-Root** zu starten.

### Voraussetzungen

* Aktives Python-Environment mit installierten Projekt-Abhängigkeiten
  (z. B. via `venv`, `conda` oder Projekt-Setup-Skript)
* Vorhandene ONNX-Datei im vorgesehenen Ordner

### Schritte

1. **Environment aktivieren** (Beispiel):

```bash
source .venv/bin/activate
```

2. **Aus dem Projekt-Root starten**:

```bash
python mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py
```

3. **Steuerung**:

* W / A / S / D / Q / E / Space zur Bewegung
* MuJoCo-Fenster muss fokussiert sein

Nach dem Start öffnet sich der MuJoCo-Viewer und das Modell kann interaktiv gesteuert werden.

---

## 1. Ziel der Livedemo

Ziel der Livedemo ist es, das trainierte Modell interaktiv vorzuführen. Die Demo basiert auf einer bereits existierenden Livedemo für ein ähnliches Modell und wurde so angepasst, dass sie mit unserem Modell kompatibel ist.
Die Demo nutzt ein **ONNX-Modell** und wird während der Laufzeit über die **Konsole mit den Tasten W, A, S, D** gesteuert.

---

## 2. Ausgangsbasis der Livedemo

Als Grundlage diente eine bestehende Livedemo aus folgendem Repository:

> **Quelle:** [play_wolfgang_joystick.py](mujoco_playground/experimental/sim2sim/play_wolfgang_joystick.py)

Die ursprüngliche Demo enthielt bereits:

* eine Konsolenanwendung,
* die Einbindung eines ONNX-Modells,
* eine einfache Tastatursteuerung (W, A, S, D).

Diese Demo wurde **kopiert** und als Basis für unser Modell verwendet. Alle Änderungen wurden innerhalb des Livedemo-Verzeichnisses vorgenommen, sodass der ursprüngliche Code unverändert erhalten bleibt.

---

## 3. Vorbereitung: ONNX-Modell erstellen und ablegen

### 3.1 Konvertierung des Modells nach ONNX

Das trainierte Modell muss zunächst in das ONNX-Format konvertiert werden.

Eine Anleitung zur ONNX-Konvertierung findet sich hier:

* [create_onnx.md](docs/create_onnx.md)

Das Ergebnis der Konvertierung ist eine Datei:

[wolves_op_policy.onnx](mujoco_playground/experimental/sim2sim/onnx/wolves_op_policy.onnx)

### 3.2 Ablage der ONNX-Datei

Die erzeugte ONNX-Datei muss im folgenden Verzeichnis abgelegt werden:

```
mujoco_playground/
├── experimental/
│   └── sim2sim
│       └── onnx
|           └── wolves_op_policy.onnx
```

Der Dateiname und Pfad werden im Livedemo-Skript referenziert.

---

## 4. Anpassungen an der Livedemo

Die Livedemo basiert auf einer bestehenden Demo-Datei, die ursprünglich für das **wolvesOP-Humanoid-Modell** entwickelt wurde und bereits eine ONNX-Policy in MuJoCo ausführt.

Die relevante Demo-Datei ist:


[play_wolfgang_joystick.py](mujoco_playground/experimental/sim2sim/play_wolfgang_joystick.py)

Der vollständige Ablauf und die notwendigen Anpassungen werden im Folgenden anhand dieser Datei erläutert.

---

### 4.1 Ablage der ONNX-Datei

Die Demo erwartet die ONNX-Datei relativ zum Speicherort der Demo-Datei:

```python
_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
```

Daraus ergibt sich folgende notwendige Ordnerstruktur:

```
experimental/sim2sim/
├── play_wolvesOP_joystick.py
└── onnx/
    └── wolves_op_policy.onnx
```

Für ein eigenes Modell muss:

* die ONNX-Datei in den Ordner `onnx/` gelegt werden
* der Dateiname im Code angepasst werden, z. B.:

```python
policy_path=(_ONNX_DIR / "model.onnx").as_posix()
```

---

### 4.2 Anpassung der Beobachtungen (Input des Modells)

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

**Wichtig:**

* Die Reihenfolge und Dimensionen müssen exakt mit dem beim Training verwendeten Beobachtungsraum übereinstimmen.
* Für ein eigenes Modell ist diese Methode der **zentrale Anpassungspunkt**.

Typische Anpassungen:

* Entfernen oder Hinzufügen von Sensorwerten
* Anpassen der Anzahl von Gelenken
* Änderung der Kommandostruktur

---

### 4.3 Anpassung der Modellausgabe (Actions)

Die Inferenz erfolgt in `get_control(...)`:

```python
onnx_input = {"obs": obs.reshape(1, -1)}
onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
```

Die Ausgabe wird anschließend direkt als Steuerkommando verwendet:

```python
data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
```

Für andere Modelle kann hier notwendig sein:

* Skalierung der Aktionen anzupassen
* Offsets zu entfernen oder zu ändern
* Aktionen auf andere Aktuatoren zu mappen

---

### 4.4 Tastatursteuerung (WASD)

Die Tastatureingaben werden über folgende Klasse verarbeitet:

```python
self._joystick = KeyboardGamepad(
    vel_scale_x=vel_scale_x,
    vel_scale_y=vel_scale_y,
    vel_scale_rot=vel_scale_rot,
)
```

Die eigentlichen WASD-Eingaben werden in:

```python
command = self._joystick.get_command()
```

abgerufen und sind Teil des Beobachtungsvektors.

---

### 4.5 MuJoCo-Initialisierung und Control-Callback

Die Funktion `load_callback(...)`:

* lädt das MuJoCo-Modell (XML + Assets)
* initialisiert Simulation und Zeitschritte
* registriert den Controller als Callback

```python
mujoco.set_mjcb_control(policy.get_control)
```

Damit wird bei jedem Simulationsschritt die Policy ausgeführt.

---

## 5. Steuerung der Livedemo

### 5.1 Tastaturbelegung

Die Steuerung erfolgt über die Konsole mit folgender Belegung:

* **W** – Bewegung nach vorne
* **A** – Bewegung nach links
* **S** – Bewegung nach hinten
* **D** – Bewegung nach rechts
* **Q** - Drehung nach links
* **E** - Drehung nach rechts

Die Tastatureingaben werden in einer Schleife abgefragt und direkt auf den aktuellen Zustand angewendet.

---

## 6. Starten der Livedemo

Die **Startdatei der Livedemo** ist das folgende Python-Skript:

[play_wolvesOP_joystick.py]([play_wolfgang_joystick.py](mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py)

Diese Datei enthält sowohl:

* die Initialisierung der MuJoCo-Simulation
* als auch die Einbindung und Ausführung der ONNX-Policy

### 6.1 Erwartete Projektstruktur

Damit die Livedemo ohne weitere Anpassungen gestartet werden kann, muss folgende Ordnerstruktur eingehalten werden:

```
experimental/sim2sim/
├── play_wolvesOP_joystick.py
└── onnx/
    └── wolves_op_policy.onnx
```

Der Ordner `onnx/` wird relativ zum Speicherort der Startdatei aufgelöst:

```python
_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
```

---

### 6.2 Start der Simulation

Die Simulation wird direkt über die Startdatei ausgeführt:

```
python experimental/sim2sim/play_wolvesOP_joystick.py
```

Beim Start werden automatisch:

1. das MuJoCo-Modell (`wolvesOP_constants.FEET_ONLY_FLAT_TERRAIN_XML`)
2. die zugehörigen Assets
3. die ONNX-Policy (`wolvesOP_policy.onnx`)
4. der Control-Callback (`OnnxController.get_control`)

initialisiert.

Anschließend öffnet sich der MuJoCo-Viewer und die Livedemo kann über die Tastatur gesteuert werden.

---

### 6.3 Anpassung für ein eigenes Modell

Um ein eigenes Modell zu verwenden, müssen mindestens folgende Stellen in der Startdatei angepasst werden:

1. **ONNX-Dateiname**

```python
policy_path=(_ONNX_DIR / "wolvesOP_policy.onnx").as_posix()
```

2. **Beobachtungsraum** in `OnnxController.get_obs(...)`

3. **Aktionsinterpretation** in `get_control(...)`

Die restliche Struktur der Startdatei kann unverändert übernommen werden.

---

## 7. Voraussetzungen

* Python-Umgebung mit ONNX Runtime
* Vorhandene ONNX-Modell-Datei im Verzeichnis `models/`
* Zugriff auf die Konsole

---

## 8. Bekannte Einschränkungen

* Die Demo ist primär für Vorführzwecke gedacht.

---

## 9. Zusammenfassung

Die Livedemo wurde auf Basis einer bestehenden Implementierung erstellt und gezielt an unser trainiertes ONNX-Modell angepasst. Durch klare Ablagepfade, minimale Codeänderungen und eine einfache Konsolensteuerung kann die Demo mit geringem Aufwand reproduziert und erweitert werden.
