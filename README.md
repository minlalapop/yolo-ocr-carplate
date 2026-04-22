## Projet YOLO + OCR pour deteter les plaques d'immatriculation ##
- utilise le dataset kaggle : https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format
- fitetuning de YOLO v8 avec la bibliothèque ultralytics
- modèle final dans /runs/detect/train-3/weights/best.pt
- notebook détaillé : car-yolo.ipynb
- utilisation de RapidOCR pour extraction de texte après avoir extrait la plaque avec YOLO

## comment tester ##
-> creer un dossier datasets/ et mettre dedans le dataset kaggle tel quel

-> executer :
```bash
python gradio-yolo.py 
```
gradio permet de tester rapidement le système avec n'importe quelle image.