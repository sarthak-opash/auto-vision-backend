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
    def __init__(self, boxes, names, orig_img):
        self.boxes = boxes
        self.names = names
        self._orig_img = orig_img
        self._orig_img_bgr = None

    @property
    def orig_img(self):
        """Lazy conversion of PIL Image to BGR numpy array."""
        if self._orig_img_bgr is None:
            import cv2
            self._orig_img_bgr = cv2.cvtColor(np.array(self._orig_img), cv2.COLOR_RGB2BGR)
        return self._orig_img_bgr

    def plot(self):
        """Draw boxes and labels on the original BGR image and return it."""
        import cv2
        img = self.orig_img.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{self.names[cls_id]} {conf:.1%}"
            
            # Draw bounding box
            color = (22, 66, 152) # Themed brand color (#984216) in BGR: (22, 66, 152)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw text label background and text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            label_y = y1 - 4 if y1 > 20 else y1 + 14
            cv2.rectangle(img, (x1, label_y - text_size[1] - 4), (x1 + text_size[0] + 4, label_y + 4), color, -1)
            cv2.putText(img, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

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
        
        # Try TensorRT, CUDA, and then CPUExecutionProvider for standard CPU fallback
        preferred_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        providers = [p for p in preferred_providers if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]
            
        # Optimize SessionOptions for maximum hardware utilization
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Dynamically allocate thread pool size for minimum inference latency
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 0
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Disable CPU memory arena to release intermediate memory allocations back to OS immediately
        opts.enable_cpu_mem_arena = False
            
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
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
        
        num_classes = len(self.names)
        box_coords = predictions[:, :4]  # Shape: (8400, 4)
        class_scores = predictions[:, 4:4+num_classes]  # Shape: (8400, num_classes)
        
        class_ids_all = np.argmax(class_scores, axis=1)
        scores_all = np.max(class_scores, axis=1)
        
        # Filter predictions by confidence threshold
        keep_mask = scores_all > conf
        
        filtered_boxes = box_coords[keep_mask]
        filtered_scores = scores_all[keep_mask]
        filtered_class_ids = class_ids_all[keep_mask]
        
        # Vectorized coordinate transformation
        if len(filtered_boxes) > 0:
            xc = filtered_boxes[:, 0]
            yc = filtered_boxes[:, 1]
            w = filtered_boxes[:, 2]
            h = filtered_boxes[:, 3]
            
            x1 = (xc - w / 2 - pad_x) / r
            y1 = (yc - h / 2 - pad_y) / r
            x2 = (xc + w / 2 - pad_x) / r
            y2 = (yc + h / 2 - pad_y) / r
            
            # Clip to original image bounds
            x1 = np.clip(x1, 0.0, float(orig_w))
            y1 = np.clip(y1, 0.0, float(orig_h))
            x2 = np.clip(x2, 0.0, float(orig_w))
            y2 = np.clip(y2, 0.0, float(orig_h))
            
            boxes_all = np.stack([x1, y1, x2, y2], axis=1)
            keep_indices = self._nms(boxes_all, filtered_scores, iou_threshold=0.45)
            
            mock_boxes = []
            for idx in keep_indices:
                mock_boxes.append(MockBox(
                    cls=int(filtered_class_ids[idx]),
                    conf=float(filtered_scores[idx]),
                    xyxy=boxes_all[idx].tolist()
                ))
        else:
            mock_boxes = []
            
        return [MockResult(boxes=mock_boxes, names=self.names, orig_img=img)]
        
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
