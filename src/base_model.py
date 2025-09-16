import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Activation, BatchNormalization

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)
            logger.info("Loaded configuration from config.yaml")
        except Exception as e:
            raise CustomException("Failed to load configuration", e)
    
    def RecommenderNet(self, n_users, n_animes):
        try:    
            # Input and embedding for users and anime
            embedding_size = self.config["model"]["embedding_size"]

            user = Input(name = "user", shape = [1])
            user_embedding = Embedding(name = "user_embedding", input_dim = n_users, output_dim = embedding_size)(user)

            anime = Input(name = "anime", shape = [1])
            anime_embedding = Embedding(name = "anime_embedding", input_dim = n_animes, output_dim = embedding_size)(anime)

            # Similarity score using dot product
            x = Dot(name = "dot_product", normalize = True, axes = 2)([user_embedding, anime_embedding])

            # Flatten the result
            x = Flatten()(x)

            # Add a dense layer
            x = Dense(1, kernel_initializer = "he_normal")(x)

            # Batch normalization
            x = BatchNormalization()(x)

            # Activation function
            x = Activation("sigmoid")(x)

            # Model
            model = Model(inputs = [user, anime], outputs = x)
            model.compile(
                loss = self.config["model"]["loss"],
                optimizer = self.config["model"]["optimizer"],
                metrics = self.config["model"]["metrics"]
            )
            logger.info("Model compiled successfully")
            return model
        except Exception as e:
            logger.error(f"Error in building the model architecture {e}")
            raise CustomException("Failed to create model", e)
