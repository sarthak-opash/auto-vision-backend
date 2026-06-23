import os
import numpy as np
import onnxruntime as ort
from PIL import Image

class MockBox:
    """Mock box class to mimic ultralytics.engine.results.Boxes."""
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = np.array([xyxy])  # So box.xyxy[0].tolist() works

class MockResult:
    """Mock result class to mimic ultralytics.engine.results.Results."""
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

class YOLOONNX:
    """
    A lightweight, drop-in replacement for ultralytics.YOLO using ONNX Runtime.
    This bypasses loading PyTorch in memory, keeping RAM usage extremely low (<150MB total).
    """
    def __init__(self, model_path: str):
        # Automatically resolve .pt path to .onnx path
        if model_path.endswith(".pt"):
            model_path = model_path.replace(".pt", ".onnx")
            
        self.model_path = model_path
        
        # Use CPUExecutionProvider for standard CPU environments (like Render free tier)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Exact class mappings from the trained YOLOv8 PyTorch weights
        self.damage_names = {
            0: 'Bodypanel-Dent', 1: 'Front-Windscreen-Damage', 2: 'Headlight-Damage', 
            3: 'Rear-windscreen-Damage', 4: 'RunningBoard-Dent', 5: 'Sidemirror-Damage', 
            6: 'Signlight-Damage', 7: 'Taillight-Damage', 8: 'bonnet-dent', 9: 'boot-dent', 
            10: 'doorouter-dent', 11: 'fender-dent', 12: 'front-bumper-dent', 13: 'pillar-dent', 
            14: 'quaterpanel-dent', 15: 'rear-bumper-dent', 16: 'roof-dent'
        }
        
        self.parts_names = {
            0: 'back_bumper', 1: 'back_door', 2: 'back_glass', 3: 'back_light', 
            4: 'front_bumper', 5: 'front_door', 6: 'front_glass', 7: 'front_light', 8: 'hood'
        }
        
        # Set self.names based on model path
        if "parts" in model_path.lower():
            self.names = self.parts_names
        else:
            self.names = self.damage_names

    def predict(self, source, conf: float = 0.25, imgsz: int = 640, verbose: bool = False):
        """
        Run inference on the image and return a list of MockResult objects.
        Mimics model.predict(source, conf, imgsz) API.
        """
        if isinstance(source, Image.Image):
            img = source
        else:
            img = Image.open(source)
            
        orig_w, orig_h = img.size
        
        # Letterbox resize (keeps aspect ratio and pads with gray)
        r = min(imgsz / orig_w, imgsz / orig_h)
        new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
        
        dw = (imgsz - new_w) / 2
        dh = (imgsz - new_h) / 2
        
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        img_padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        
        pad_x = int(round(dw - 0.1))
        pad_y = int(round(dh - 0.1))
        img_padded.paste(img_resized, (pad_x, pad_y))
        
        # Preprocessing: Normalize to [0, 1] and reshape to BCHW
        img_data = np.array(img_padded).astype(np.float32) / 255.0
        
        # Handle grayscale or alpha channels if any
        if len(img_data.shape) == 2:
            img_data = np.stack([img_data]*3, axis=-1)
        elif img_data.shape[2] == 4:
            img_data = img_data[:, :, :3]
            
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        
        # Run Session
        outputs = self.session.run([self.output_name], {self.input_name: img_data})
        output_tensor = outputs[0]
        
        # Postprocessing: Decode prediction boxes
        # Output shape is (1, 4 + num_classes, 8400)
        # Transpose to (8400, 4 + num_classes)
        predictions = np.transpose(output_tensor[0], (1, 0))
        
        boxes = []
        scores = []
        class_ids = []
        num_classes = len(self.names)
        
        for pred in predictions:
            box_scores = pred[4:4+num_classes]
            class_id = np.argmax(box_scores)
            score = box_scores[class_id]
            
            if score > conf:
                xc, yc, w, h = pred[0:4]
                
                # Convert back to original unpadded coordinate space
                x1 = (xc - w / 2 - pad_x) / r
                y1 = (yc - h / 2 - pad_y) / r
                x2 = (xc + w / 2 - pad_x) / r
                y2 = (yc + h / 2 - pad_y) / r
                
                # Clip to original image bounds
                x1 = max(0.0, min(x1, float(orig_w)))
                y1 = max(0.0, min(y1, float(orig_h)))
                x2 = max(0.0, min(x2, float(orig_w)))
                y2 = max(0.0, min(y2, float(orig_h)))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
                
        # Non-Maximum Suppression (NMS)
        keep_indices = self._nms(np.array(boxes), np.array(scores), iou_threshold=0.45)
        
        mock_boxes = []
        for idx in keep_indices:
            mock_boxes.append(MockBox(
                cls=class_ids[idx],
                conf=scores[idx],
                xyxy=boxes[idx]
            ))
            
        return [MockResult(boxes=mock_boxes, names=self.names)]
        
    def _nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep
