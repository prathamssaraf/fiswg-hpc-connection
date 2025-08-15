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
            
            return output_text, prediction
            
        except Exception as e:
            logger.error(f"Error analyzing image pair: {e}")
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
        """Parse model output to extract same/different person prediction from detailed forensic analysis"""
        output_lower = output_text.lower()
        
        # Primary: Look for the final conclusion statement (most reliable)
        if "conclusion:" in output_lower:
            conclusion_part = output_lower.split("conclusion:")[-1]
            if "same person" in conclusion_part or "same individual" in conclusion_part:
                return True
            elif "different people" in conclusion_part or "different person" in conclusion_part or "different individuals" in conclusion_part:
                return False
        
        # Secondary: Look for Expert Determination section
        if "expert determination:" in output_lower:
            determination_part = output_lower.split("expert determination:")[-1]
            if "same individual" in determination_part or "same person" in determination_part:
                return True
            elif "different individuals" in determination_part or "different people" in determination_part:
                return False
        
        # Tertiary: Look for forensic conclusion section
        if "forensic conclusion" in output_lower:
            forensic_part = output_lower.split("forensic conclusion")[-1]
            if "same person" in forensic_part or "same individual" in forensic_part:
                return True
            elif "different people" in forensic_part or "different individuals" in forensic_part:
                return False
        
        # Fallback: Enhanced keyword analysis with weighted scoring
        same_indicators = [
            ("same person", 5), ("same individual", 5), ("identical", 3), 
            ("match", 2), ("correlate", 2), ("consistent", 1), ("similar", 1)
        ]
        different_indicators = [
            ("different people", 5), ("different person", 5), ("different individuals", 5),
            ("not the same", 4), ("distinct", 3), ("inconsistent", 2), ("contradict", 2)
        ]
        
        same_score = sum(weight for phrase, weight in same_indicators if phrase in output_lower)
        different_score = sum(weight for phrase, weight in different_indicators if phrase in output_lower)
        
        if same_score > different_score:
            return True
        elif different_score > same_score:
            return False
        else:
            # Enhanced fallback: look for positive/negative language patterns
            positive_patterns = ["support identity", "confirm identity", "establish identity"]
            negative_patterns = ["exclude identity", "rule out", "cannot confirm"]
            
            has_positive = any(pattern in output_lower for pattern in positive_patterns)
            has_negative = any(pattern in output_lower for pattern in negative_patterns)
            
            if has_positive and not has_negative:
                return True
            elif has_negative and not has_positive:
                return False
            else:
                # Final default to different if completely unclear
                logger.warning(f"Unclear prediction in forensic output: {output_text[:300]}...")
                return False