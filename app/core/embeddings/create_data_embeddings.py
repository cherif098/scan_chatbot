import os
import uuid

import pandas as pd
from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models

from app.core.embeddings.embeddings_utils import convert_text_to_embeddings, convert_image_to_embeddings, TEXT_MODEL_NAME, IMAGE_MODEL_NAME

# Utiliser des chemins relatifs par rapport au répertoire du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')

def create_uuid_from_image_id(image_id):
    NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(NAMESPACE_UUID, image_id))

def create_embeddings(collection_name):
    # Vérifier si le dossier data existe, sinon le créer
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        os.makedirs(os.path.join(DATA_PATH, 'images'))
        # Créer un fichier captions.txt d'exemple
        with open(os.path.join(DATA_PATH, 'captions.txt'), 'w') as f:
            f.write('example_image\tThis is an example caption.')
        print(f"Created data directory at {DATA_PATH}")
        print("Please add your images to the 'images' folder and update captions.txt before running again.")
        return QdrantClient(":memory:")

    # Read captions txt data
    path = os.path.join(DATA_PATH, 'captions.txt')
    caption_df = pd.read_csv(path, sep='\t', header=None, names=['image_id', 'caption'])

    # Read images
    images_path = os.path.join(DATA_PATH, 'images')
    image_directory = os.listdir(images_path)

    # Filter out images that are not in the captions
    images = []
    for image in image_directory:
        if image.split('.')[0] in caption_df['image_id'].values:
            images.append(image)

    # Create image_id, caption, image_path list of dictionaries
    image_docs = []
    for image in images:
        image_id = image.split('.')[0]
        caption = caption_df[caption_df['image_id'] == image_id]['caption'].values[0]
        image_path = os.path.join(images_path, image)
        image_docs.append({'image_id': image_id, 'caption': caption, 'image_path': image_path})

    if not image_docs:
        print("No matching images found. Please ensure your images are in the 'images' folder and properly referenced in captions.txt")
        return QdrantClient(":memory:")

    # Convert text to embeddings using Fastembed
    captions = [doc['caption'] for doc in image_docs]
    embeddings = convert_text_to_embeddings(captions)
    for idx, embedding in enumerate(embeddings):
        image_docs[idx]['caption_embedding'] = embedding

    # Convert image to embeddings using CLIP
    image_embeddings = convert_image_to_embeddings([doc['image_path'] for doc in image_docs])
    for idx, embedding in enumerate(image_embeddings):
        image_docs[idx]['image_embedding'] = embedding

    # Save the embeddings to vector database
    client = QdrantClient(":memory:")

    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    text_embeddings_size = text_model._get_model_description(TEXT_MODEL_NAME)["dim"]

    image_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)
    image_embeddings_size = image_model._get_model_description(IMAGE_MODEL_NAME)["dim"]

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=image_embeddings_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )
    
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=create_uuid_from_image_id(doc['image_id']),
                vector={
                    "text": doc['caption_embedding'],
                    "image": doc['image_embedding'],
                },
                payload={
                    "image_id": doc['image_id'],
                    "caption": doc['caption'],
                    "image_path": doc['image_path']
                }
            )
            for doc in image_docs
        ]
    )
    return client