import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.ai_model import AIModel
from app.utils.file_handler import save_upload_file, cleanup_file
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
ai_model = AIModel()

@router.post("/process")
async def process_query(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        logger.info("Starting process_query")
        logger.info(f"Received query: {query}")
        
        image_path = None
        if image:
            logger.info(f"Processing image: {image.filename}")
            try:
                image_path = await save_upload_file(image)
                logger.info(f"Image saved to: {image_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

        try:
            logger.info("Calling AI model")
            response = await ai_model.process_query(query, image_path)
            logger.info("AI model response received")
        except Exception as e:
            logger.error(f"Error in AI model processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in AI processing: {str(e)}")

        # Nettoyer le fichier temporaire
        if image_path:
            try:
                cleanup_file(image_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

        return {"response": response}

    except Exception as e:
        logger.error(f"Unhandled error in process_query: {str(e)}")
        if image_path:
            try:
                cleanup_file(image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))