import os
import logging
from app.core.gpt.gpt_utils import GPTClient

logger = logging.getLogger(__name__)

class MultimodalRAGSystem:
    def __init__(self):
        self.gpt_client = GPTClient()

    def process_query(self, query, query_image_path=None):
        # Vérification de l'existence de l'image
        if query_image_path and not os.path.exists(query_image_path):
            logger.warning(f"Image not found at {query_image_path}")
            return "Error: The provided image could not be found."

        try:
            # Appel direct à GPT-4 avec l'image uploadée
            gpt_response = self.gpt_client.query(query, [], query_image_path)
            return self.gpt_client.process_response(gpt_response)
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return f"Error processing the query: {str(e)}"
