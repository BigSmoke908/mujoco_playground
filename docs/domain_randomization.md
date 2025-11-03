# Domain-Randomization

## Warum?

Echte Roboter verhalten sich nie wie das perfekt Modell -> mit leicht verschiedenen Robotern trainieren hilft das auszugleichen


## Grundlagen

Zu einem Locomotion Environment gehört immer ein Randomizer, um die spezifische Domain-Randomization durchzuführen. Welcher Randomizer zu welchem Environment gehört wird in der [__init__.py](../mujoco_playground/_src/locomotion/__init__.py) von dem `locomotion-Package` angegeben. Beispiel:
```py
...
from mujoco_playground._src.locomotion.wolves_op import randomize as wolvesop_randomize
...

_randomizer = {
    ...,
    "WolvesOPJoystickFlatTerrain": wolvesop_randomize.domain_randomize,
    "WolvesOPJoystickRoughTerrain": wolvesop_randomize.domain_randomize,
}

...
```

Die Konvention ist, dass jedes Locomotion-Environment eine [randomize.py](../mujoco_playground/_src/locomotion/wolves_op/randomize.py) hat, in der die Bot-spezifische Domain-Randomization durchgeführt wird.


## Was macht unsere Randomization aktuell

 - Random Gain: Factor mit dem Motor-Output multipiziert wird -> simuliert leicht verschiedene Motoren (bei unserer Parametrisierung extrem stark)
 - Random Friction: Bodenhaften variieren
 - Random Armature: Trägheit der Motoren ("wie viel Kraft wird für eine Beschleunigung gebraucht")
 - Random Damping: Dämpfung der Motoren ("wie stark bremst der Motor eine Bewegung aus")
 - Random Massen: die Masse aller Bauteile sind leicht skaliert
 - Random COMs: die COMs (Center-Of-Mass/Schwerpunkte) der Bauteile sind leicht verschoben
 - Random qpos0: Startposition bei jeder Episode ist leicht verschieden


## Was könnte man noch einbauen

 - Random Pertubations: jede Rotationsaxe ist zufällig ein kleines bisschen verdreht
 - Random Backlash: jeder Motor hat auch wenn er eine fest Position anfährt ein gewisses Spiel (kann noch ein bisschen gedreht werden) (scheint nicht nativ in Mujoco umgesetzt zu sein->https://mujoco.readthedocs.io/en/stable/modeling.html#backlash)
 - 
