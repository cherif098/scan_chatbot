import os
import uuid
from fastapi import UploadFile
from app.config import settings

async def save_upload_file(file: UploadFile) -> str:
    """Sauvegarde le fichier uploadé et retourne son chemin"""
    # Créer le dossier uploads s'il n'existe pas
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Générer un nom de fichier unique
    file_extension = os.path.splitext(file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    
    # Sauvegarder le fichier
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

def cleanup_file(file_path: str):
    """Supprime le fichier temporaire"""
    if os.path.exists(file_path):
        os.remove(file_path)
