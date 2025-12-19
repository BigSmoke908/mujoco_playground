# ONNX-Datei aus trainierter Policy erzeugen
> Eine ONNX Datei wird benötigt um die Policy später auszuführen. Das Ausführen der Policy wird [hier](./Livedemo.md) beschrieben.

## als Teil vom Training

 - die ONNX Datei kann im Anschluss an das Training direkt automatisch erzeugt werden
 - hierfür muss bei dem Aufruf für das Training (`python learning/train_jax_ppo.py ...`) ein Ordner für die Erzeugung der ONNX übergeben werden
 - bsp.: `python learning/train_jax_ppo.py ... --model=model`
 - der Ordner wird automatisch neben den checkpoint-Order (`logs/WolvesOPJoystickFlatTerrain-..../checkpoints/`) platziert und enthält die ONNX mit der trainierten Policy

### vollständiger Beispielaufruf

`python learning/train_jax_ppo.py --env_name=WolvesOPJoystickFlatTerrain --model=model`

## nach dem Training

 - die ONNX Datei kann nach dem abgeschlossenen Training jederzeit aus den checkpoints erzeugt werden
 - hierfür das [convert_to_onnx.py](../learning/utils/convert_to_onnx.py) Skript wie folgt aufrufen:
 
 `python learning/utils/convert_to_onnx.py --checkpoint={CHECKPOINT} --output={OUTPUT} --env_name={ENV_NAME}`

### vollständiger Beispielaufruf
`python learning/utils/convert_to_onnx.py --checkpoint=docs/working_policy/WolvesOPJoystickFlatTerrain-20251207-163905/checkpoint/000151388160 --output=mujoco_playground/experimental/sim2sim/onnx/wolves_op_policy.onnx --env_name=WolvesOPJoystickFlatTerrain`


### Parameter erklärt

 - CHECKPOINT: Pfad zu dem Checkpoint, bsp.: `logs/WolvesOPJoystickFlatTerrain-20251109-183828/checkpoints/000151388160/`
 - OUTPUT (optional): Pfad zu der generierten ONNX-Datei, bsp.: `wolvesOP_policy.onnx`
 - ENV_NAME (optional): Name von der verwendeten Mujoco-Umgebung, bsp.: `WolvesOPJoystickFlatTerrain`

