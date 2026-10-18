"""Fine-tune YOLO object detector ONLY after annotating images in YOLO format."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='configs/objects.yaml')
    p.add_argument('--weights', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    from ultralytics import YOLO
    dataset = Path(args.data)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    print('WARNING: YOLO object labels must exist; video-level violation labels cannot train bounding boxes.')
    m = YOLO(args.weights)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            project='runs', name='bank_objects', exist_ok=True)
    print('Weights: runs/bank_objects/weights/best.pt')
if __name__ == '__main__': main()
