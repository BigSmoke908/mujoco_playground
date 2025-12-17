# Reward-Funktionen – WolvesOP Joystick Environment

Hinweis: Die Reward-Funktion basiert auf einer Kopie des bestehenden [Wolfgang-Environments](../mujoco_playground/_src/locomotion/wolfgang/). Daher lassen sich nicht alle Designentscheidungen im Detail begründen.

## Zweck der Reward-Funktion


Die Reward-Funktion bewertet das Verhalten des Agenten während der Simulation im Mujoco-Playground. Sie liefert in jedem Zeitschritt ein skalierbares Feedback-Signal für das Reinforcement Learning und steuert, welche Bewegungsmuster als erwünscht oder unerwünscht gelten.

Ziel ist es, einen stabilen, energieeffizienten und befehlsgetreuen Gang des humanoiden Roboters zu erlernen. Die Reward-Berechnung erfolgt vollständig im Zeitschritt und basiert auf einer gewichteten Summe mehrerer Teil-Rewards und Kosten.

Die vollständige Implementierung der Reward-Funktion befindet sich in der Datei Joystick.py. Dort werden alle Teil-Rewards berechnet, skaliert und zu einem Gesamtreward zusammengeführt. Die zentrale Logik ist in der Methode _get_reward(...) gekapselt, welche pro Zeitschritt alle einzelnen Reward- und Kostenkomponenten auswertet und zurückgibt.
Über die reward_config lassen sich zudem die Skalierungsfaktoren konfigurieren (siehe Abschnitt Grundprinzip).

---

## Grundprinzip

Das Reward-System folgt drei Prinzipien:

- **Tracking**: Der Agent folgt vorgegebenen linearen und angularen Geschwindigkeitskommandos.
- **Stabilität**: Instabile und physikalisch unplausible Zustände werden bestraft.
- **Effizienz**: Energieverbrauch und abrupte Bewegungen werden reduziert.

Alle Teil-Rewards werden über Skalierungsfaktoren gewichtet und anschließend zeitlich integriert. Die Skalierungsfaktoren sind in der reward_config innerhalb der [Joystick.py](../mujoco_playground/_src/locomotion/wolves_op/joystick.py) definiert und können dort angepasst werden.

---

## Gesamt-Reward

In jedem Zeitschritt werden alle einzelnen Reward-Terme berechnet, skaliert und aufsummiert. Der resultierende Reward wird mit der Simulationszeit `dt` multipliziert und anschließend begrenzt:

- positive Terme wirken als Belohnung
- negative Terme wirken als Kosten

---

## Tracking-Rewards

### `tracking_lin_vel`

Bewertet die Abweichung zwischen gewünschter und gemessener lokaler linearer Geschwindigkeit (x/y).

### `tracking_ang_vel`

Bewertet die Abweichung zwischen gewünschter und gemessener Yaw-Winkelgeschwindigkeit.

---

## Basis- und Stabilitätskosten

### `lin_vel_z`

Bestrafung vertikaler Rumpfgeschwindigkeit zur Vermeidung von Springen oder Fallen.

### `ang_vel_xy`

Bestrafung von Rotationen um die x- und y-Achse als Maß für Instabilität.

### `orientation`

Bestrafung von Abweichungen der Rumpforientierung von der aufrechten Haltung.

### `base_height`

Quadratische Abweichung zwischen aktueller und gewünschter Rumpfhöhe.

---

## Energie- und Glättungskosten

### `torques`

Summe der absoluten Aktuatordrehmomente zur Reduktion mechanischer Belastung.

### `energy`

Energieverbrauch als Produkt aus Gelenkgeschwindigkeit und Aktuatorkraft.

### `action_rate`

Quadratische Bestrafung starker Änderungen zwischen aufeinanderfolgenden Aktionen.

---

## Fuß- und Gang-bezogene Rewards

### `feet_slip`

Bestrafung von lateraler Bewegung der Füße während Bodenkontakt (Rutschen).

### `feet_clearance`

Bewertet Abweichungen der Fußhöhe von einer gewünschten maximalen Schwunghöhe, gewichtet mit der Fußgeschwindigkeit.

### `feet_height`

Bewertet die maximale Fußhöhe während der Schwungphase beim ersten Bodenkontakt.

### `feet_air_time`

Belohnt eine angemessene Dauer der Schwungphase, sofern ein Bewegungsbefehl vorliegt.

### `feet_phase`

Vergleicht die tatsächliche Fußhöhe mit einer phasenabhängigen Soll-Trajektorie zur Förderung eines konsistenten Gangzyklus.

---

## Haltungs- und Gelenkkosten

### `joint_deviation_hip`

Bestrafung von Abweichungen der Hüftgelenke von der Neutralpose bei seitlichen Bewegungen.

### `joint_deviation_knee`

Bestrafung von Abweichungen der Kniegelenke von der Referenzpose.

### `dof_pos_limits`

Bestrafung von Überschreitungen weicher Gelenkgrenzen.

### `pose`

Gewichtete quadratische Abweichung aller Gelenke von der Default-Pose zur Sicherstellung einer natürlichen Grundhaltung.

---

## Sonstige Rewards

### `alive`

Konstante positive Belohnung, solange keine Terminierungsbedingung eintritt.

### `stand_still`

Bestrafung von Abweichungen von der Neutralpose bei nahezu null Bewegungsbefehlen.

### `termination`

Einmalige Strafzahlung bei Sturz oder Instabilität.

---

## Hilfsfunktionen

Zur Abbildung von Fehlerwerten werden allgemeine Toleranz- und Sigmoidfunktionen verwendet. Diese ermöglichen weiche Übergänge anstelle harter Schwellwerte und sorgen für stabile Gradienten während des Trainings.

