# ONNX-Datei aus trainierter Policy erzeugen
> Eine ONNX Datei (+eventuell andere Artefakte) wird benötigt um die Policy später auszuführen

## als Teil vom Training

 - die ONNX Datei (+andere Artefakte) kann im Anschluss an das Training direkt automatisch erzeugt werden
 - hierfür muss bei dem Aufruf für das Training (`python learning/train_jax_ppo.py ...`) ein Ordner für die Erzeugung der Artefakte übergeben werden
 - bsp.: `python learning/train_jax_ppo.py ... --model=model`
 - der Ordner wird automatisch neben den checkpoint-Order (`logs/WolvesOPJoystickFlatTerrain-..../checkpoints/`) platziert und enthält alle Artefakte

## nach dem Training

 - die ONNX Datei (+andere Artefakte) kann nach dem abgeschlossenen Training jederzeit aus den checkpoints erzeugt werden
 - hierfür das [convert_to_onnx.py](../learning/utils/convert_to_onnx.py) Skript wie folgt aufrufen:
 
 `python learning/utils/convert_to_onnx.py --checkpoint={CHECKPOINT} --output={OUTPUT} --env_name={ENV_NAME}`


### Parameter erklärt

 - CHECKPOINT: Pfad zu dem Checkpoint, bsp.: `logs/WolvesOPJoystickFlatTerrain-20251109-183828/checkpoints/000151388160/`
 - OUTPUT (optional): Pfad zu der generierten ONNX-Datei, bsp.: `wolvesOP_policy.onnx`
 - ENV_NAME (optional): Name von der verwendeten Mujoco-Umgebung, bsp.: `WolvesOPJoystickFlatTerrain`

