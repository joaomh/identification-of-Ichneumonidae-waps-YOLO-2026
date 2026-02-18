import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tqdm import tqdm
# --- CONFIGURATION ---
OUTPUT_DIR = Path('reports/HiResCAM_Visualizations')
SOURCE_DIR = TEST_DIR
USE_HIRESCAM = True

# --- WRAPPER CLASS ---
class YOLOCAMWrapper(nn.Module):
    def __init__(self, yolo_model):
        super(YOLOCAMWrapper, self).__init__()
        self.model = yolo_model

    def forward(self, x):
        result = self.model(x)
        if isinstance(result, tuple):
            return result[0]
        return result

def run_annotated_gradcam():
    print(f"Loading Model: {MODEL_PATH}")
    # Load YOLO
    hub_model = YOLO(MODEL_PATH)
    
    # Get class names from the model (e.g., {0: 'Braconidae', 1: 'Colletidae', ...})
    class_names = hub_model.names
    print(f"Model Classes: {class_names}")

    # Force Gradients ON
    for param in hub_model.model.parameters():
        param.requires_grad = True
        
    # Setup Wrapper
    model = YOLOCAMWrapper(hub_model.model)
    model.eval()
    
    # Target Layer
    target_layers = [model.model.model[-2]]
    
    # Setup CAM
    cam_cls = HiResCAM if USE_HIRESCAM else GradCAM
    cam = cam_cls(model=model, target_layers=target_layers)

    # Find Images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.tif', '*.tiff','*.TIF','*.TIFF']
    all_images = []
    for ext in extensions:
        all_images.extend(list(Path(SOURCE_DIR).rglob(ext)))
    
    print(f"Found {len(all_images)} images to process.")

    # --- PROCESSING LOOP ---
    for img_path in tqdm(all_images, desc="Analyzing"):
        try:
            # 1. Get Ground Truth from Folder Name
            # e.g., .../val/Braconidae/img1.jpg -> true_label = "Braconidae"
            true_label = img_path.parent.name
            
            # 2. Load & Preprocess
            img_pil = Image.open(img_path).convert('RGB')
            original_w, original_h = img_pil.size
            
            img_resized = np.array(img_pil.resize((224, 224)))
            rgb_img_float = np.float32(img_resized) / 255.0
            
            input_tensor = preprocess_image(rgb_img_float, mean=[0,0,0], std=[1,1,1])
            input_tensor.requires_grad = True

            # 3. Run Inference (Get Prediction)
            # We run a forward pass to get logits
            logits = model(input_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = class_names[pred_idx]
            
            # Check correctness
            is_correct = (pred_label == true_label)
            
            # 4. Generate Heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # Upscale
            heatmap_high_res = cv2.resize(grayscale_cam, (original_w, original_h))
            rgb_img_high_res = np.float32(img_pil) / 255.0
            visualization = show_cam_on_image(rgb_img_high_res, heatmap_high_res, use_rgb=True)
            
            # Convert to BGR for OpenCV drawing
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

            # 5. Add Text Annotation
            # Green if correct, Red if wrong
            color = (0, 255, 0) if is_correct else (0, 0, 255) 
            status_text = "CORRECT" if is_correct else "WRONG"
            
            text = f"True: {true_label} | Pred: {pred_label} ({status_text})"
            
            # Draw black background rectangle for text readability
            #cv2.rectangle(visualization_bgr, (0, 0), (original_w, 150), (0, 0, 0), -1)
            
            # Draw text (Scale font based on image size)
            font_scale = max(2.0, original_w / 1000.0) 
            thickness = max(2, int(font_scale * 2))
            
            #cv2.putText(visualization_bgr, text, (20, 100), 
                        #cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # 6. Save File
            # Prefix filename with status for easy sorting
            prefix = "CORRECT_" if is_correct else "WRONG_"
            new_filename = prefix + img_path.name
            
            # Save structure: output/Braconidae/WRONG_img1.jpg
            relative_folder = img_path.parent.name
            save_folder = Path(OUTPUT_DIR) / relative_folder
            save_folder.mkdir(parents=True, exist_ok=True)
            
            save_path = save_folder / new_filename
            cv2.imwrite(str(save_path), visualization_bgr)

        except Exception as e:
            print(f"\nSkipped {img_path.name}: {e}")
            continue

    print(f"\nDone! Check folder: {OUTPUT_DIR}")

if __name__ == '__main__':
    run_annotated_gradcam()