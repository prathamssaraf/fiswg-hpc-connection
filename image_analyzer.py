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
        
        # Debug: Log key sections found
        sections_found = []
        if "conclusion:" in output_lower:
            sections_found.append("conclusion")
        if "expert determination:" in output_lower:
            sections_found.append("expert_determination")
        if "forensic conclusion" in output_lower:
            sections_found.append("forensic_conclusion")
        
        logger.debug(f"Sections found: {sections_found}")
        
        # Primary: Look for the final conclusion statement (most reliable)
        if "conclusion:" in output_lower:
            conclusion_part = output_lower.split("conclusion:")[-1]
            logger.debug(f"Conclusion part: {conclusion_part[:200]}")
            
            # More comprehensive same person patterns
            same_patterns = ["same person", "same individual", "identical person", "same identity"]
            different_patterns = ["different people", "different person", "different individuals", 
                                "different identity", "not the same person", "distinct individuals"]
            
            for pattern in same_patterns:
                if pattern in conclusion_part:
                    logger.debug(f"Found same pattern: {pattern}")
                    return True
            for pattern in different_patterns:
                if pattern in conclusion_part:
                    logger.debug(f"Found different pattern: {pattern}")
                    return False
        
        # Secondary: Look for Expert Determination section
        if "expert determination:" in output_lower:
            determination_part = output_lower.split("expert determination:")[-1]
            logger.debug(f"Expert determination part: {determination_part[:200]}")
            
            if any(pattern in determination_part for pattern in ["same individual", "same person", "identical"]):
                return True
            elif any(pattern in determination_part for pattern in ["different individuals", "different people", "not the same"]):
                return False
        
        # Tertiary: Look for forensic conclusion section
        if "forensic conclusion" in output_lower:
            forensic_part = output_lower.split("forensic conclusion")[-1]
            logger.debug(f"Forensic conclusion part: {forensic_part[:200]}")
            
            if any(pattern in forensic_part for pattern in ["same person", "same individual", "identical"]):
                return True
            elif any(pattern in forensic_part for pattern in ["different people", "different individuals"]):
                return False
        
        # Enhanced fallback: Look for any conclusive statements
        conclusive_same = [
            "are the same person", "is the same person", "same identity",
            "match confirmed", "positive identification", "identity confirmed"
        ]
        conclusive_different = [
            "are different people", "not the same person", "different identities",
            "no match", "negative identification", "identity excluded"
        ]
        
        for pattern in conclusive_same:
            if pattern in output_lower:
                logger.debug(f"Found conclusive same pattern: {pattern}")
                return True
        for pattern in conclusive_different:
            if pattern in output_lower:
                logger.debug(f"Found conclusive different pattern: {pattern}")
                return False
        
        # Weighted keyword analysis (improved)
        same_indicators = [
            ("same person", 10), ("same individual", 10), ("identical", 5), 
            ("match", 3), ("correlate", 3), ("consistent", 2), ("similar", 1)
        ]
        different_indicators = [
            ("different people", 10), ("different person", 10), ("different individuals", 10),
            ("not the same", 8), ("distinct", 5), ("inconsistent", 3), ("contradict", 3)
        ]
        
        same_score = sum(weight for phrase, weight in same_indicators if phrase in output_lower)
        different_score = sum(weight for phrase, weight in different_indicators if phrase in output_lower)
        
        logger.debug(f"Same score: {same_score}, Different score: {different_score}")
        
        if same_score > different_score and same_score > 0:
            return True
        elif different_score > same_score and different_score > 0:
            return False
        else:
            # Final fallback: Look for general positive/negative language
            positive_patterns = ["support identity", "confirm identity", "establish identity", "yes", "positive"]
            negative_patterns = ["exclude identity", "rule out", "cannot confirm", "no", "negative"]
            
            has_positive = any(pattern in output_lower for pattern in positive_patterns)
            has_negative = any(pattern in output_lower for pattern in negative_patterns)
            
            if has_positive and not has_negative:
                logger.debug("Using positive language fallback")
                return True
            elif has_negative and not has_positive:
                logger.debug("Using negative language fallback")
                return False
            else:
                # Ultimate fallback: analyze overall sentiment
                logger.warning(f"Unable to parse prediction clearly from output: {output_text[:200]}...")
                # Default to False (different) for unclear cases
                return False