import torch_directml
from controlnet_aux import MLSDdetector
from PIL import Image
import os

def extract_room_skeleton(image_path, output_path):
    # 1. Configurazione Device (AMD)
    device = torch_directml.device()
    
    # 2. Caricamento Immagine
    # La GenAI lavora meglio con immagini quadrate o multipli di 64
    input_image = Image.open(image_path).convert("RGB")
    
    # 3. Inizializzazione del Detector MLSD
    # MLSD = Mobile Line Segment Detection (ottimo per interni)
    print("Caricamento modello MLSD...")
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    
    # 4. Elaborazione
    # thr_v e thr_d sono le soglie di rilevamento per linee verticali e distanti
    print("Estrazione linee architettoniche in corso...")
    detected_map = mlsd(input_image, thr_v=0.1, thr_d=0.1)
    
    # 5. Salvataggio
    detected_map.save(output_path)
    print(f"Mappa salvata con successo in: {output_path}")

if __name__ == "__main__":
    # Assicurati di avere una foto chiamata 'stanza.jpg' nella cartella
    # o cambia il nome qui sotto
    test_image = "data/input/stanza.jpg" 
    if os.path.exists(test_image):
        extract_room_skeleton(test_image, "data/output/stanza_skeleton.png")
    else:
        print(f"Errore: Carica una foto chiamata '{test_image}' nella cartella!")