# Documentation: WolvesOP Environment

# Basis:
 - Das Environment von WolvesOP basiert auf einer Kopie des Wolfgang Environments.
 - Da wir entsprechend den Namen unseres neuen Environments geändert haben, mussten in allen Dateien die Imports und Verweise auf Dateien des Environment geändert werden, damit diese wieder korrekt zueinander passen

 Konkret haben wir diese Dateien angepasst:
 - locomotion/ _init_.py -> neues Environment in _envs, _cfgs, _randomizer einfügen
 - wolves_op/ base.py -> imports, def get_assets()
 - wolves_op/ convert_folder.py -> vollständig selber erstellt
 - wolves_op/ demo.py -> ???
 - wolves_op/ export_policy.py -> vollständig selber erstellt
 - wolves_op/ joystick.py -> import, siehe XML integration
 - wolves_op/ randomize.py -> ???
 - wolves_op/ wolvesop_constants.py -> Rootpath XML, siehe XML Integration

 - Vor der Arbeit an einem neuen Environment sollte sichergestellt werden, dass ein Training
 im Wolfgang Environment erfolgreich durchgeführt werden kann, um technische Funktionalität
 auf der eigenen Maschine zu prüfen.

# XML Integration:
 Sobald eine funktionale XML erstellt und in das neue Environment eingefügt wurde, haben wir einige Änderungen Vorgenommen um es an unseren Robi anzupassen.

 - Der Pfad unserer XML muss im Environment angepasst werden.
    -> Dafür waren jeweils Änderungen in 'wolvesop_constants.py' (line 21f) und in der base.py (line 34).

 - Außerdem muss sichergestellt werden, dass alle Bezeichner in der XML mit denen im Environment übereinstimmen.
    -> Also sowohl Joint Namen als auch Sensoren

# Joint Übersicht (Wolfgang Environment):

 - LL = Leg Left
 - LR = Leg Right

| Joint Name   | Stichwort Bewegung (Funktion)                     | Erklärung (Anatomie)                                               |
|--------------|--------------------------------------------------|--------------------------------------------------------------------|
| LL_HR / LR_HR | Hüft Roll                                       | Rotation des Oberschenkels um die Längsachse (Roll-Bewegung)       |
| LL_HAA / LR_HAA | Hüft Adduktion/Abduktion                      | Bewegung des Beins von der Mittellinie weg/hin (Seitwärtsbewegung)|
| LL_HFE / LR_HFE | Hüft Flexion/Extension                         | Beugen und Strecken des Hüftgelenks (Vor- und Rückwärtsbewegung)  |
| LL_KFE / LR_KFE | Knie Flexion/Extension                         | Beugen und Strecken des Knies                                      |
| LL_FFE / LR_FFE | Fuß Flexion/Extension (Plantar-/Dorsalflexion)| Bewegung des Fußes nach unten/oben (Knöchel-Bewegung)             |
| LL_FAA / LR_FAA | Fuß Adduktion/Abduktion (Inversion/Eversion) | Seitwärtsbewegung des Fußes (nach innen/außen kippen)             |




# Vorgehensweise bei Anpassungen:
 - Training starten und entstehende Fehlermeldungen prüfen.
 - Bezeichner in der XML oder im Environment so anpassen, dass sie zueinander passen.
 - Wiederhole bis es funktioniert xD
 - Die meisten Anpassungen betreffen die Dateien 'wolvesop_constants' und 'Joystick.py'.

-----
-----
-----
-----

# WolvesOP Projektübersicht



| Dateiname | Kurzbeschreibung | Rolle im System |
| :--- | :--- | :--- |
| `base.py` | **Basisumgebungsklasse** für WolvesOP-Roboter. | Definiert die `WolvesOPEnv`-Klasse, lädt MuJoCo XML-Dateien, konvertiert sie in das JAX-kompatible **MJX**-Format und stellt grundlegende **Sensor-Lese-Methoden** bereit. |
| `wolvesop_constants.py` | **Konstanten** und **Pfaddefinitionen**. | Enthält Pfade zu den MuJoCo-Modellen und -Aufgaben (z. B. `flat_terrain.xml`), sowie Bezeichnungen für wichtige **Sensoren, Körper und Geometrien**. |
| `joystick.py` | **Aufgabenspezifische Trainingsumgebung**. | Implementiert die `Joystick`-Klasse als Unterklasse von `WolvesOPEnv`. Enthält **Reset-Logik**, Berechnung der Beobachtungen (`_get_obs`), Simulationsschritte (`step`), **Domänen-Randomisierung** und das **Belohnungssystem**. |
| `randomize.py` | **Domänen-Randomisierung**. | Enthält `domain_randomize`, das **statische MuJoCo-Modellparameter** wie Masse, Reibung oder Dämpfung randomisiert, um Policies robuster gegenüber Sim2Real-Unterschieden zu machen. |
| `convert_folder.py` | **Orbax → Pickle Konverter**. | Hilfsskript zum Laden eines **Orbax-Checkpoints** und Export in eine portable **`.pkl`-Datei**. Dient als Zwischenschritt vor weiterem Export. |
| `export_policy.py` | **Policy-Export nach ONNX**. | Lädt eine Policy aus `.pkl` oder Orbax, erstellt ein äquivalentes **TensorFlow Keras-Modell**, überträgt die Gewichte und exportiert die Policy anschließend als **ONNX-Modell**. |

---

## Detaillierte Zusammenhänge

### 1. Basis & Konfiguration  
**(`base.py`, `wolvesop_constants.py`)**

- `wolvesop_constants.py` liefert die **XML-Pfade** zu Robotermodellen und Aufgaben.
- `base.py` verwendet diese Pfade, um die **Grundumgebung** zu initialisieren.
- Außerdem stellt `base.py` Methoden bereit, um:
  - Sensordaten auszulesen,
  - Zustände abzurufen,
  - MuJoCo-Modelle in **MJX** umzuwandeln.

---

### 2. Aufgabe & Simulationslogik  
**(`joystick.py`)**

- Erbt von `base.WolvesOPEnv`.
- Definiert:
  - **Belohnungsfunktion** (`_get_reward`)
  - **Simulationsschritte** (`step`)
  - **Zustandsermittlung** (`_get_obs`)
  - **Sampling von Commands/Zielgeschwindigkeiten**
  - **Randomisierte Push-Störungen** während der Simulation  
- Diese Datei beschreibt also die **komplette Roboter-Aufgabe**, die im Training genutzt wird.

---

### 3. Policy-Training & Robustheit  
**(`randomize.py`, `joystick.py`)**

- `randomize.py` ist zuständig für **statische Parameteränderungen** (Masse, Reibung, Dämpfung).
- `joystick.py` wendet **dynamische Randomisierung** an (Push-Forces während des `step`).
- Zusammengenommen erhöhen beide Mechanismen die **Sim2Real-Robustheit** der trainierten Policy.

---

### 4. Export-Pipeline  
**(`convert_folder.py`, `export_policy.py`)**

1. Die Umgebung (`joystick.py`) erzeugt nach dem Training einen **Orbax-Checkpoint**.  
2. `convert_folder.py` kann diesen in eine `.pkl`-Datei umwandeln.  
3. `export_policy.py`:
   - lädt die Parameter,
   - baut ein äquivalentes **TensorFlow Policy-Netzwerk**,
   - überträgt die Gewichte,
   - speichert das fertige Modell als **ONNX**, das universell einsetzbar ist (z. B. für Embedded-Systeme).

---


