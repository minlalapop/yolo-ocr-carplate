## Projet YOLO + OCR pour deteter les plaques d'immatriculation ##
- utilise le dataset kaggle : https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format
- fitetuning de YOLO v8 avec la bibliothèque ultralytics
- modèle final dans /runs/detect/train-3/weights/best.pt
- notebook détaillé : car-yolo.ipynb
- utilisation de RapidOCR pour extraction de texte après avoir extrait la plaque avec YOLO

## contenu 
- voir notebook : car-yolo.ipynb
- voir evalutation/metrics/graphics : runs/train-3/

## comment tester ##
- installer requirements 
```bash
pip install -r requirements.txt
```
- unzip le dataset kaggle dans datasets/
- tester YOLO+OCR facilement avec gradio
```bash
python gradio-yolo.py 
```
NB : gradio permet de tester rapidement le système avec n'importe quelle image.