# Portierung der Umgebung: Von Wolfgang zu WolvesOP

Diese Dokumentation beschreibt den Prozess der Erstellung der **WolvesOP**-Umgebung innerhalb des `mujoco_playground`. Als Basis diente die bereits existierende Umgebung **Wolfgang**.

## 1. Verzeichnisstruktur und Duplizierung

Der erste Schritt bestand darin, eine solide Basis für den neuen Roboter zu schaffen. Da WolvesOP ähnliche kinematische Eigenschaften oder Anforderungen wie Wolfgang aufweist, wurde der bestehende Code als Vorlage genutzt.

* **Vorgehen:** Der komplette Ordner `mujoco_playground/_src/locomotion/wolfgang` wurde kopiert und als `mujoco_playground/_src/locomotion/wolves_op` eingefügt.
* **Ziel:** Nutzung der bestehenden Infrastruktur (Joystick-Logik, Reward-Funktionen, MJX-Setup), um Entwicklungszeit zu sparen.

## 2. Anpassung der Python-Dateien

Nach dem Kopieren mussten die Python-Dateien angepasst werden, um auf die neuen Pfade, Klassennamen und spezifischen Anforderungen des WolvesOP-Modells zu verweisen.

### A. Konstanten (`wolvesop_constants.py`)

Die Datei wurde von `wolfgang_constants.py` in `wolvesop_constants.py` umbenannt.

* **Pfadanpassung:** Die Variable `ROOT_PATH` wurde geändert, damit Assets und XML-Dateien im neuen `wolves_op`-Verzeichnis gesucht werden.
    ```python
    ...
    ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "wolves_op"
    ...
    ```
* **Sensorik & Geometrie:** Die Definitionen für Füße (`FEET_SITES`, `FEET_GEOMS`) und Sensoren wurden zunächst beibehalten, da die Namen in der XML (z. B. `l_foot`, `r_foot`) an das Basis-Environment angepasst wurden.

### B. Basis-Umgebung (`base.py`)

Hier wurden die grundlegenden Funktionen zum Laden des Modells und der Assets implementiert.

1.  **Klassen-Umbenennung:**
    Die Hauptklasse wurde von `WolfgangEnv` in `WolvesOPEnv` umbenannt, um Konflikte zu vermeiden und die Zugehörigkeit klarzustellen.

2.  **Erweitertes Asset-Loading (`get_assets`):**
    
    In dem Standard-Mujoco-Playground werden stl-Dateien über die Mujoco-Menagerie verwaltet. Um den Prozess für uns zu vereinfachen, haben wir das Asset-Loading Script angepasst um so die Menagerie zu umgehen:
    
    ```python
    # Hinzugefügt in WolvesOP:
    stl_path = consts.ROOT_PATH / "xmls" / "stls"
    for f in stl_path.glob("*.stl"):
        if f.is_file():
            # Das Asset-Dictionary benötigt den Key so, wie er im XML referenziert wird
            assets[f"stls/{f.name}"] = f.read_bytes()
    ```

3.  **MJX Initialisierung:**
    Im Konstruktor (`__init__`) wird das Mujoco-Modell geladen und anschließend mittels `mjx.put_model(self._mj_model)` in ein MJX-Modell konvertiert, um die GPU-Beschleunigung via JAX zu ermöglichen.

### C. Joystick-Logik (`joystick.py`)

Die Datei `joystick.py` steuert das Verhalten des Roboters und berechnet die Belohnungen (Rewards) für das Reinforcement Learning.

1.  **Imports und Vererbung:**
    Die Klasse `Joystick` erbt nun von `wolvesop_base.WolvesOPEnv` statt von der Wolfgang-Basisklasse. Alle Imports wurden auf `wolves_op` umgebogen.

2.  **Anpassung der Konfiguration (`default_config`):**
    Die Konfiguration (z. B. `lin_vel_x`, `lin_vel_y`) wurde übernommen. Es wurden Kommentare hinzugefügt, um Bereiche für zukünftiges Tuning zu markieren (z. B. Anpassung der Geschwindigkeitsgrenzen für Tests).


## 3. Integration & Registrierung

> Damit das Trainingsskript `train_jax_ppo.py` die neue Umgebung finden und nutzen kann, müssen zwei Komponenten angepasst werden.

### Registrierung (`mujoco_playground/_src/locomotion/__init__.py`)

> hier werden alle Environments inklusive Environment, Default-Config und Domain-Randomizer angegeben. Zu beachten ist, dass der Environmentname (hier "WolvesOPJoystickFlatTerrain") an allen Stellen übereinstimmt.

Die Komponente importieren:
```python
...
from mujoco_playground._src.locomotion.wolves_op import joystick as wolvesop_joystick
from mujoco_playground._src.locomotion.wolves_op import randomize as wolvesop_randomize
...
```

Das Environment verlinken:
    ```python
    _envs = {
        ...
        "WolvesOPJoystickFlatTerrain": functools.partial(
            wolvesop_joystick.Joystick, task="flat_terrain"
        ),
    }
    ```


Ein Config mit default Werten für das Training verlinken. Diese Config bezieht sich auf spezifische Config-Werte für hauptsächlich die Rewards:
    ```python
    _cfgs = {
        ...
        "WolvesOPJoystickFlatTerrain": wolvesop_joystick.default_config,
    }
    ```

Den Domain-Randomizer für das Environment angeben:
    ```python
    _randomizer = {
        ...
        "WolvesOPJoystickFlatTerrain": wolvesop_randomize.domain_randomize,
    }
    ```


### Trainings-Parameter (`mujoco_playground/config/locomotion_params.py`)

> Hier werden die Hyperparameter für das PPO-Training definiert

**Hinzugefügter Block:**
```python
elif env_name in (
    "WolvesOPJoystickFlatTerrain",
    "WolvesOPJoystickRoughTerrain",
):
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 15
    rl_config.clipping_epsilon = 0.2
    # ... weitere Parameter ...
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        # ...
    )
    ```
