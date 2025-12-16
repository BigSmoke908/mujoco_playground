# Dokumentation zur Livedemo

## Quick Guide (Kurzstart)

Dieser Abschnitt beschreibt die minimal notwendigen Schritte, um die Livedemo **direkt aus dem Projekt-Root** zu starten.

### Voraussetzungen

* Aktives Python-Environment mit installierten Projekt-Abhängigkeiten (z. B. via `venv`, `conda` oder Projekt-Setup-Skript)
* Vorhandene ONNX-Datei im vorgesehenen Ordner

### Schritte

1. **Environment aktivieren** (Beispiel):

```bash
source .venv/bin/activate
```

2. **Livedemo starten**:

```bash
python mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py
```

3. **Steuerung**:

* W / A / S / D – Bewegung
* Q / E – Rotation
* Space – Reset
* MuJoCo-Fenster muss fokussiert sein

Nach dem Start öffnet sich der MuJoCo-Viewer und das Modell kann interaktiv gesteuert werden.

---

## 1. Ziel der Livedemo

Ziel der Livedemo ist es, das trainierte Modell interaktiv vorzuführen. Die Demo basiert auf einer bestehenden Livedemo für ein ähnliches Modell und wurde so angepasst, dass sie mit dem aktuellen Modell kompatibel ist. Die Steuerung erfolgt während der Laufzeit über die Tastatur (W, A, S, D, Q, E).

---

## 2. Ausgangsbasis

Als Grundlage diente eine bestehende Livedemo aus folgendem Repository:

> **Quelle:** [play_wolfgang_joystick.py](mujoco_playground/experimental/sim2sim/play_wolfgang_joystick.py)

Die ursprüngliche Demo enthielt bereits:

* Konsolenanwendung
* Einbindung eines ONNX-Modells
* Einfache Tastatursteuerung

Die Datei wurde kopiert und für das neue Modell angepasst. Der Originalcode bleibt unverändert erhalten.

---

## 3. Vorbereitung: ONNX-Modell

### 3.1 Konvertierung nach ONNX

Das trainierte Modell muss zunächst in das ONNX-Format konvertiert werden. Eine Anleitung befindet sich hier:

* [create_onnx.md](docs/create_onnx.md)

Das Ergebnis ist eine ONNX-Datei:

[wolves_op_policy.onnx](mujoco_playground/experimental/sim2sim/onnx/wolves_op_policy.onnx)

### 3.2 Ablage der ONNX-Datei

Die ONNX-Datei muss im folgenden Verzeichnis liegen:

```
mujoco_playground/
└── experimental/
    └── sim2sim/
        └── onnx/
            └── wolves_op_policy.onnx
```

Der Pfad wird relativ zur Startdatei aufgelöst.

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
policy_path = (_ONNX_DIR / "model.onnx").as_posix()
```

---

### 4.2 Beobachtungen (Modelleingang)

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

### 4.3 Aktionen (Modellausgang)

Die Kommunikation erfolgt in `get_control(...)`:

```python
onnx_input = {"obs": obs.reshape(1, -1)}
onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
```

Die Ausgabe wird als Steuerkommando verwendet:

```python
data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
```

Je nach Modell können hier Skalierung, Offsets oder Aktuator-Zuordnung angepasst werden.

---

### 4.4 Tastatursteuerung

Die Tastatureingaben werden über einen `KeyboardGamepad` verarbeitet:

```python
self._joystick = KeyboardGamepad(
    vel_scale_x=vel_scale_x,
    vel_scale_y=vel_scale_y,
    vel_scale_rot=vel_scale_rot,
)
```

Die daraus resultierenden Kommandos fließen als Teil der Beobachtung in das Modell ein:

```python
command = self._joystick.get_command()
```

---

### 4.5 MuJoCo-Initialisierung

Beim Start werden:

* das MuJoCo-Modell (XML + Assets)
* die Simulation
* die ONNX-Policy

initialisiert. Der Controller wird als Callback registriert:

```python
mujoco.set_mjcb_control(policy.get_control)
```

Damit wird bei jedem Simulationsschritt die Policy ausgeführt.

---

## 5. Start der Livedemo

Die Simulation wird über folgende Datei gestartet:

```
python experimental/sim2sim/play_wolvesOP_joystick.py
```

Nach dem Start öffnet sich der MuJoCo-Viewer und das Modell kann interaktiv gesteuert werden.

---

## 6. Anpassung für eigene Modelle

Für ein eigenes Modell müssen in der Startdatei mindestens folgende Punkte angepasst werden:

1. **ONNX-Dateiname**
2. **Beobachtungsraum** in `get_obs(...)`
3. **Aktionsinterpretation** in `get_control(...)`

Die restliche Struktur kann unverändert übernommen werden.

---

## 7. Voraussetzungen

* Python-Umgebung mit ONNX Runtime
* Vorhandene ONNX-Modell-Datei im angegebenen Verzeichnis
* Tastaturfokus auf dem MuJoCo-Fenster

---

## 8. Einschränkungen

* Die Demo ist primär für Vorführ- und Testzwecke gedacht.

---

## 9. Zusammenfassung

Die Livedemo ermöglicht eine interaktive Vorführung eines trainierten ONNX-Modells in MuJoCo. Durch klare Ordnerstrukturen, eine einfache Tastatursteuerung und minimale Anpassungen lässt sich die Demo leicht reproduzieren und auf andere Modelle übertragen.
