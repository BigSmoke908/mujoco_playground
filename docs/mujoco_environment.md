# Documentation: WolvesOP Environment

## Grundstruktur von dem Environemnt:
> das Environment von WolvesOP basiert auf einer Kopie des Wolfgang Environments aus dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#)


### Dateien in den Environment

 - [wolves_op/base.py](../mujoco_playground/_src/locomotion/wolves_op/base.py): Basis für die Umgebung, hier wird das [Mujoco-XML-Modell](mujoco_model_documentation.md) geladen und einige Utilfunktionen werden definiert 
 - [wolves_op/joystick.py](../mujoco_playground/_src/locomotion/wolves_op/joystick.py): die "Aufgabe" für das Reinforcement-Learning, hier wird ist die [Reward-Function](./reward_functions.md) und -Konfiguration für das Training definiert
 - [wolves_op/randomize.py](../mujoco_playground/_src/locomotion/wolves_op/randomize.py): hier wird die [Domain-Randomization](./domain_randomization.md) für das Training durchgeführt
 - [wolves_op/wolvesop_constants.py](../mujoco_playground/_src/locomotion/wolves_op/wolvesop_constants.py): Konstanten für die Addressierung von Bauteilen in der XML werden definiert (wichtig: nicht alle Konstanten sind hier definiert -> ein refactoring wäre wahrscheinlich sinnvoll)
 - [wolves_op/xmls/](../mujoco_playground/_src/locomotion/wolves_op/xmls/): hier befindet sich das [Mujoco-XML-Modell](./mujoco_model_documentation.md)
 - [wolves_op/xmls/stls](../mujoco_playground/_src/locomotion/wolves_op/xmls/stls/): hier befinden sich die in dem [Mujoco-XML-Modell](./mujoco_model_documentation.md) referenzierten .stl-Meshes


### Model-Integration

Das [Mujoco-XML-Modell](mujoco_model_documentation.md) muss in der [wolvesop_constants.py](../mujoco_playground/_src/locomotion/wolves_op/wolvesop_constants.py)in den beiden Variablen `FEET_ONLY_FLAT_TERRAIN_XML` und `FEET_ONLY_ROUGH_TERRAIN_XML` angegeben werden, damit diese im Training später verwendet werden können. Da in dem Environment aus dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) einige Werte konstant definiert sind (Actuator-Namen, Sensor-Namen, etc.) muss das verwendet Modell die in dieser [Dokumention](./mujoco_model_documentation.md) angegebenen Anforderungen erfüllen.


## WolvesOP-Environment

> Hier wird die Erzeugung von [wolvesop-Environment](../mujoco_playground/_src/locomotion/wolves_op/) beschrieben. Environments für weitere Roboter können äquivalent erstellt werden.

### Grundlage

Als Grundlage für das Environment dient das [wolfgang-Environment](../mujoco_playground/_src/locomotion/wolfgang/) aus dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#). Für das neue Environment wurde der gesamte Ordner kopiert und in `wolves_op` umbenannt. Um von in diesem Projekt gemachten Änderungen (bsp. vereinfachtes Asset-Loading) zu profitieren, kann anschließend auch das [wolves_op-Environment](../mujoco_playground/_src/locomotion/wolves_op/) als Basis verwendet werden.

### Konsistente Namensgebung

Der Environemt-Name ist in mehrere Dateien enthalten und muss für die Konsistenz an diesen Stellen angepasst werden:

### [wolvesop_constants.py](../mujoco_playground/_src/locomotion/wolves_op/wolvesop_constants.py)

Diese Datei wird von `wolfgang_constants.py` in `wolvesop_constants.py` umbenannt. Außerdem muss der `ROOT_PATH` des Environments angepasst werden:

```python
ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "wolves_op"
```

### [base.py](../mujoco_playground/_src/locomotion/wolves_op/base.py)

Hier muss der Import der Konstanten an das neue Environment angepasst werden:

```python
from mujoco_playground._src.locomotion.wolves_op import wolvesop_constants as consts
```

und der Asset-Pfad in der `get_assets()`-Funktion muss auch an das neue Environment angepasst werden:

```python
path = mjx_env.MENAGERIE_PATH / "wolvesop"
```


Nach diesen beiden Änderungen muss die gesamte Klasse nach dem neuen Environemnt benannt werden:

```python
class WolvesOPEnv(mjx_env.MjxEnv):
```


### [joystick.py](../mujoco_playground/_src/locomotion/wolves_op/joystick.py)

In dieser Klasse müssen beiden Umbenennungen aus den ersten beiden Schritten auch durchgeführt werden. Das betrifft diese beiden Imports:

```python
from mujoco_playground._src.locomotion.wolves_op import base as wolvesop_base
from mujoco_playground._src.locomotion.wolves_op import wolvesop_constants as consts
```

und diese Vererbung:
```python
class Joystick(wolvesop_base.WolvesOPEnv):
```

### randomize.py](../mujoco_playground/_src/locomotion/wolves_op/randomize.py)

In dieser Datei wurden Änderungen für eine [erweiterte Domain-Randomization](./domain_randomization.md) durchgeführt. Für ein Aufsetzen von neuen Environments sind hier keine Änderungen notwendig.


### Registrierung (`mujoco_playground/_src/locomotion/__init__.py`)

> Neben der Anpassung von dem eigenen Environment muss noch die `registry` von dem Mujoco-Playground angepasst werden, damit das Environment im Training verwendet werden kann.

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


Ein Config mit default Werten für das Training verlinken. Diese Config bezieht sich auf spezifische Config-Werte für hauptsächlich die [Rewards](./reward_functions.md):
    ```python
    _cfgs = {
        ...
        "WolvesOPJoystickFlatTerrain": wolvesop_joystick.default_config,
    }
    ```

Den [Domain-Randomizer](./domain_randomization.md) für das Environment angeben:
    ```python
    _randomizer = {
        ...
        "WolvesOPJoystickFlatTerrain": wolvesop_randomize.domain_randomize,
    }
    ```


### Trainings-Parameter (`mujoco_playground/config/locomotion_params.py`)

Hier werden die Hyperparameter für das PPO-Training definiert. Da Standardwerte existieren sind die Anpassungen nicht unbedingt notwendig. Dennoch wurden die Werte aus dem [Basisprojekt](https://github.com/bit-bots/mujoco_playground#) übernommen.


```python
def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
   ...
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

