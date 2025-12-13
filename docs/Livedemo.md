# Dokumentation zur Livedemo

## 1. Ziel der Livedemo

Ziel der Livedemo ist es, das trainierte Modell interaktiv vorzuführen. Die Demo basiert auf einer bereits existierenden Livedemo für ein ähnliches Modell und wurde so angepasst, dass sie mit unserem Modell kompatibel ist. Die Demo benötigt eine **ONNX-Datei** und während der Laufzeit über die **Konsole mit den Tasten W, A, S, D** gesteuert.

## 2. Ausgangsbasis

Als Grundlage diente eine bestehende Livedemo für ein ähnliches Modell. Diese enthielt bereits:

* eine grundlegende Konsolenanwendung,
* die Einbindung eines Modells,
* eine einfache Steuerlogik.

Die vorhandene Demo wurde analysiert und anschließend gezielt an die Anforderungen unseres Modells angepasst.

## 3. Anpassungen für das neue Modell

Die folgenden Anpassungen wurden vorgenommen:

### 3.1 Modellintegration (ONNX)

* Das trainierte Modell liegt im **ONNX-Format (.onnx)** vor.
* Die bestehende Modell-Ladefunktion wurde so angepasst, dass sie die neue ONNX-Datei lädt.
* Pfade und Dateinamen wurden entsprechend geändert.
* Die Ein- und Ausgabeformate des Modells wurden an die Spezifikation des neuen Modells angepasst.

### 3.2 Anpassung der Eingaben

* Die Eingabedaten für das Modell wurden an die erwartete Struktur des neuen Modells angepasst.
* Falls notwendig, wurden Vorverarbeitungsschritte (z. B. Normalisierung oder Skalierung) übernommen oder modifiziert.

### 3.3 Anpassung der Ausgaben

* Die Ausgaben des Modells werden ausgewertet und in Steuerungsbefehle bzw. Zustände übersetzt.
* Die Logik zur Interpretation der Modellausgabe wurde an das neue Modell angepasst.

## 4. Steuerung der Livedemo

Die Steuerung der Demo erfolgt über die Tastatur in der Konsole.

### 4.1 Tastaturbelegung

* **W**: Bewegung nach vorne
* **A**: Drehung nach links
* **S**: Bewegung nach hinten
* **D**: Drehung nach rechts

Die Eingaben werden in Echtzeit verarbeitet und beeinflussen das Verhalten des Modells bzw. der Simulation.

### 4.2 Konsoleninteraktion

* Die Konsole dient als Hauptschnittstelle für die Benutzerinteraktion.
* Tasteneingaben werden kontinuierlich abgefragt.
* Der aktuelle Zustand oder relevante Ausgaben werden optional in der Konsole ausgegeben (z. B. Debug-Informationen).

## 5. Voraussetzungen

* Vorhandene ONNX-Modell-Datei.
* Zugriff auf die Konsole/Tastatureingaben.

## 6. Bekannte Einschränkungen

* Die Demo ist primär für Vorführzwecke gedacht und nicht für den produktiven Einsatz optimiert.

## 7. Zusammenfassung

Die Livedemo basiert auf einer bestehenden Implementierung und wurde erfolgreich an ein neues Modell angepasst. Durch die Nutzung einer ONNX-Datei und einer einfachen Konsolensteuerung kann das Modell interaktiv demonstriert und sein Verhalten anschaulich präsentiert werden.
