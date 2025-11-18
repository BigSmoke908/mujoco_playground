- um die simulierung zu starten: `python mujoco_playground/experimental/sim2sim/play_wolvesOP_joystick.py`
- um aus log eine pkl datei zu erstellen: `python mujoco_playground/_src/locomotion/wolves_op/convert_folder.py`
- um onnx datei aus einer pkl datei zu erstellen: `python mujoco_playground/eexperimental/x02_walking/convert_to_onnx.py --checkpoint policy.pkl --output wolves_op_policy.onnx --env-name WolvesOPJoystickFlatTerrain    `


- noch nicht funktionsfähig, aber ist daszu gedacht aus den Trainingsdaten in logs, eine onnx datei direkt zu erstellen aber es gibt noch fehler: `python mujoco_playground/_src/locomotion/wolves_op/export_policy.py --checkpoint logs/WolvesOPJoystickFlatTerrain-20251104-075648/checkpoints/000151388160  --output mujoco_playground/experimental/sim2sim/onnx/wolves_op_policy.onnx --env-name WolvesOPJoystickFlatTerrain`