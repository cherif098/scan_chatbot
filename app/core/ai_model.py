import logging
from app.config import settings
import os
from app.core.multimodal_rag_system import MultimodalRAGSystem

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self):
        try:
            logger.info("Initializing MultimodalRAGSystem")
            self.model = MultimodalRAGSystem()
            logger.info("MultimodalRAGSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MultimodalRAGSystem: {str(e)}")
            raise

    async def process_query(self, query: str, image_path: str = None) -> str:
        try:
            logger.info(f"Processing query: {query}")
            logger.info(f"Image path: {image_path}")
            
            # Vérifier si l'image existe
            if image_path and not os.path.exists(image_path):
                logger.error(f"Image file not found at: {image_path}")
                raise Exception("Image file not found")

            response = self.model.process_query(query, image_path)
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            raise Exception(f"Error processing query: {str(e)}")