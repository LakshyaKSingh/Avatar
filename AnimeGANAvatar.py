import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
import time
import sys
import logging
import absl.logging
import warnings

# Suppress MediaPipe and TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")

class AnimeGANAvatar:
    def __init__(self, model_name="face_paint_512_v2.pt"):
        Path("captured_images").mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing on device: {self.device}")
        
        self.model = None
        self.face_mesh = None
        
        self._load_model(model_name)
        self._initialize_face_mesh()

    def _load_model(self, model_name):
        """Improved model loading with better error handling"""
        try:
            model_path = Path("models/bryandlee_animegan2-pytorch_main")
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
                
            # Add the model directory to Python path
            sys.path.insert(0, str(model_path))
            
            # Import the Generator class from the model
            try:
                from model import Generator
            except ImportError:
                # If the import fails, try to import from the correct location
                from animegan2_pytorch.model import Generator
            
            # Load with proper device mapping
            self.model = Generator()
            weights_path = model_path / "weights" / model_name
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
                
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {str(e)}")
            raise
        except ImportError as e:
            logging.error(f"Failed to import model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise

    def _initialize_face_mesh(self):
        """MediaPipe initialization with suppressed warnings"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_eye_regions(self, frame):
        """Extract and enhance both eye regions."""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            left_eye, right_eye = None, None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Left and right eye landmarks
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]

                    left_eye = [
                        (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                        for idx, landmark in enumerate(face_landmarks.landmark)
                        if idx in left_eye_indices
                    ]
                    right_eye = [
                        (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                        for idx, landmark in enumerate(face_landmarks.landmark)
                        if idx in right_eye_indices
                    ]

            return left_eye, right_eye
        except Exception as e:
            print(f"Error extracting eye regions: {e}")
            return None, None

    def enhance_eye(self, frame, eye_points):
        """Apply enhancements to a single eye region."""
        try:
            if eye_points is not None:
                # Create a mask for the eye
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(eye_points, np.int32)], 255)

                # Extract and enhance the eye region
                eye_region = cv2.bitwise_and(frame, frame, mask=mask)
                eye_region = cv2.GaussianBlur(eye_region, (15, 15), 30)  # Enhance eyes
                enhanced_eye = cv2.addWeighted(frame, 0.7, eye_region, 0.3, 0)

                # Combine back into the original frame
                frame[mask > 0] = enhanced_eye[mask > 0]
            return frame
        except Exception as e:
            print(f"Error enhancing eye: {e}")
            return frame

    def process_image(self, image):
        """Main processing pipeline with error handling"""
        try:
            # Convert input to RGB numpy array
            img = self._validate_input(image)
            
            # Process image
            img = self._enhance_eyes(img)
            anime_img = self._apply_anime_style(img)
            
            return self._postprocess(anime_img)
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return image

    def _validate_input(self, image):
        """Validate and convert input image"""
        if isinstance(image, bytes):
            arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = image.copy()
            
        if img.size == 0:
            raise ValueError("Empty image received")
            
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _enhance_eyes(self, img):
        """Enhance eyes in the image"""
        left_eye, right_eye = self.extract_eye_regions(img)
        if left_eye is not None and right_eye is not None:
            img = self.enhance_eye(img, left_eye)
            img = self.enhance_eye(img, right_eye)
        return img

    def _apply_anime_style(self, img):
        """Apply anime style to the image"""
        # Convert image to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Apply the model
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        
        return output

    def _postprocess(self, img):
        """Post-process the image"""
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def process_frame(self, frame):
        """Process a single video frame."""
        anime_frame = self.process_image(frame)

        try:
            # Resize frame directly to 512x512 for model input
            img = cv2.resize(frame, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to tensor for model inference
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            img = img.to(self.device) / 127.5 - 1.0

            # Generate anime style
            with torch.no_grad():
                out = self.model(img)

            # Convert back to image
            out = out[0].permute(1, 2, 0).cpu().numpy()
            out = (out + 1) * 127.5
            out = np.clip(out, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            return out

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def process_video(self):
        """Capture video and apply processing."""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            anime_frame = self.process_frame(frame)

            cv2.imshow('Original', frame)
            cv2.imshow('Anime Style', anime_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                original_path = f"captured_images/original_{timestamp}.jpg"
                anime_path = f"captured_images/anime_{timestamp}.jpg"
                cv2.imwrite(original_path, frame)
                cv2.imwrite(anime_path, anime_frame)
                print(f"Images captured at timestamp: {timestamp}")
                print(f"Original image saved at: {original_path}")
                print(f"Anime image saved at: {anime_path}")

        cap.release()
        cv2.destroyAllWindows()

    def list_saved_images(self):
        """List all saved images in the captured_images directory."""
        return list(Path("captured_images").glob("*.jpg"))


if __name__ == "__main__":
    try:
        avatar = AnimeGANAvatar()
        avatar.process_video()
    except Exception as e:
        print(f"Error: {e}")