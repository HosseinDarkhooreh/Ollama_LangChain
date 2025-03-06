import ollama
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Utils:

    

    def extract_model_names() -> Tuple[str, ...]:
        """
        Extract model names from the provided models information.

        """
        logger.info("Extracting model names from models_info")
        try:

            models_info = ollama.list()
            print(models_info)
            # The new response format returns a list of Model objects
            if hasattr(models_info, "models"):
                # Extract model names from the Model objects
                model_names = tuple(model.model for model in models_info.models)
            else:
                # Fallback for any other format
                model_names = tuple()
                
            logger.info(f"Extracted model names: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Error extracting model names: {e}")
            return tuple()
   
    

        

    
