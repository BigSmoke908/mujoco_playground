import orbax.checkpoint as ocp
import pathlib
import pickle
import os

# ==========================================
# HIER DEINEN PFAD EINTRAGEN
# Beispiel: "logs/WolvesOPJoystickFlatTerrain-2025.../checkpoints/1000000"
checkpoint_path = "/home/ubuntu/albaraa/mujoco_playground/logs/WolvesOPJoystickFlatTerrain-20251104-075648/checkpoints/000151388160" 
# ==========================================

def main():
    # Pfad sicherstellen
    path = pathlib.Path(checkpoint_path).resolve()
    
    if not path.exists():
        print(f"FEHLER: Der Ordner wurde nicht gefunden:\n{path}")
        return

    print(f"Lade Orbax-Checkpoint aus Ordner: {path} ...")
    
    # Orbax Checkpointer initialisieren und Ordner laden
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        params = checkpointer.restore(path)
    except Exception as e:
        print(f"Fehler beim Laden mit Orbax: {e}")
        return

    print("Checkpoint erfolgreich geladen. Speichere als policy.pkl ...")

    # Als einfache .pkl Datei speichern
    output_file = "policy.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(params, f)

    print(f"✅ Fertig! Die Datei liegt hier: {os.path.abspath(output_file)}")
    print("Diese 'policy.pkl' kannst du jetzt deinem Kollegen geben.")

if __name__ == "__main__":
    main()