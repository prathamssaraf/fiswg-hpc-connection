"""
Image Analyzer for Facial Comparison
Handles image preprocessing and analysis using Qwen2.5-VL model
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import Tuple
from forensic_prompts import FORENSIC_PROMPT
from config import MODEL_GENERATION_CONFIG

# Import qwen_vl_utils for vision processing
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback function if qwen_vl_utils not available
    def process_vision_info(messages):
        """Helper function to process vision information from messages"""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                for content in message["content"]:
                    if content["type"] == "image":
                        image_inputs.append(content["image"])
                    elif content["type"] == "video":
                        video_inputs.append(content["video"])
        
        return image_inputs, video_inputs

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Handles image analysis and facial comparison using Qwen2.5-VL"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.forensic_prompt = FORENSIC_PROMPT

    def analyze_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[str, bool]:
        """Analyze a pair of images and return prediction"""
        try:
            # Preprocess images
            pil_img1, pil_img2 = self._preprocess_images(img1, img2)
            
            # Prepare messages for the model
            messages = self._prepare_messages(pil_img1, pil_img2)
            
            # Process with model
            output_text = self._process_with_model(messages)
            
            # Parse prediction from output
            prediction = self._parse_prediction(output_text)
            
            # Debug logging for first few pairs
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 3:  # Log first 3 outputs for debugging
                logger.info(f"DEBUG - Model output sample {self._debug_count}:")
                logger.info(f"Output length: {len(output_text)} characters")
                logger.info(f"First 500 chars: {output_text[:500]}")
                logger.info(f"Last 500 chars: {output_text[-500:]}")
                logger.info(f"Prediction: {prediction}")
            
            return output_text, prediction
            
        except Exception as e:
            logger.error(f"Error analyzing image pair: {e}")
            # Check if this is a Triton cache manager error that we can ignore
            if "TRITON_CACHE_MANAGER" in str(e):
                logger.warning("Triton cache manager error - attempting to continue processing")
                # Try to continue with a default response
                return "Unable to process due to Triton cache error", False
            return f"Error: {str(e)}", False

    def _preprocess_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Image.Image, Image.Image]:
        """Convert numpy arrays to PIL Images with proper formatting"""
        # Convert to uint8 if needed
        if img1.dtype == np.float64 or img1.dtype == np.float32:
            img1 = (img1 * 255).astype(np.uint8)
        if img2.dtype == np.float64 or img2.dtype == np.float32:
            img2 = (img2 * 255).astype(np.uint8)
            
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        
        return pil_img1, pil_img2

    def _prepare_messages(self, pil_img1: Image.Image, pil_img2: Image.Image) -> list:
        """Prepare messages for the model"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img1},
                    {"type": "image", "image": pil_img2},
                    {"type": "text", "text": self.forensic_prompt}
                ]
            }
        ]

    def _process_with_model(self, messages: list) -> str:
        """Process messages with the loaded model"""
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model not loaded. Call model_manager.load_model() first.")
        
        # Prepare inputs
        text = self.model_manager.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.model_manager.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model_manager.model.generate(
                **inputs,
                **MODEL_GENERATION_CONFIG
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.model_manager.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text


    def _parse_prediction(self, output_text: str) -> bool:
        """Parse simple YES/NO model output"""
        output_text = output_text.strip().upper()
        
        # Direct YES/NO parsing
        if output_text.startswith('YES'):
            logger.debug("Found YES - same person")
            return True
        elif output_text.startswith('NO'):
            logger.debug("Found NO - different people")
            return False
        elif 'YES' in output_text and 'NO' not in output_text:
            logger.debug("Found YES in output - same person")
            return True
        elif 'NO' in output_text and 'YES' not in output_text:
            logger.debug("Found NO in output - different people")
            return False
        else:
            # Log the unexpected output for debugging
            logger.warning(f"Unexpected output format (not YES/NO): {output_text[:100]}")
            # Default to False (different) for unclear cases
            return False