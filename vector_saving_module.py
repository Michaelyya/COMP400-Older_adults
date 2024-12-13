import json
import faiss
import numpy as np
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import logging
import pickle

# Load environment variables
load_dotenv()

@dataclass
class Exercise:
    id: str
    name: str
    category: str
    difficulty: str
    intensity: str
    suitable_conditions: List[str]
    contraindications: List[str]
    description: str
    benefits: List[str]
    original_data: Dict

class ExerciseVectorDB:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.exercises: List[Exercise] = []
        self.vector_store = None
        self.dimension = 1536  # OpenAI embedding dimension
        self.index_file = "exercise_index.faiss"
        self.exercise_file = "exercise_data.pkl"
        
    def save_index(self) -> None:
        """Save FAISS index and exercise data to disk"""
        try:
            # Save FAISS index
            if self.vector_store is not None:
                faiss.write_index(self.vector_store, self.index_file)
            
            # Save exercise data
            with open(self.exercise_file, 'wb') as f:
                pickle.dump(self.exercises, f)
                
            logging.info("Saved index and exercise data to disk")
            
        except Exception as e:
            logging.error(f"Error saving index: {e}")
            raise

    def load_index(self) -> bool:
        """Load FAISS index and exercise data from disk"""
        try:
            # Check if files exist
            if not os.path.exists(self.index_file) or not os.path.exists(self.exercise_file):
                logging.info("No existing index found")
                return False
            
            # Load FAISS index
            self.vector_store = faiss.read_index(self.index_file)
            
            # Load exercise data
            with open(self.exercise_file, 'rb') as f:
                self.exercises = pickle.load(f)
                
            logging.info("Loaded index and exercise data from disk")
            return True
            
        except Exception as e:
            logging.error(f"Error loading index: {e}")
            return False

    def load_exercises(self, file_path: str) -> None:
        """Load exercises from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.exercises = []
            for exercise in data['exercises']:
                self.exercises.append(
                    Exercise(
                        id=exercise['id'],
                        name=exercise['name'],
                        category=exercise['category'],
                        difficulty=exercise['difficulty'],
                        intensity=exercise['intensity'],
                        suitable_conditions=exercise['suitable_conditions'],
                        contraindications=exercise['contraindications'],
                        description=exercise['description'],
                        benefits=exercise['benefits'],
                        original_data=exercise
                    )
                )
            logging.info(f"Loaded {len(self.exercises)} exercises")
        except Exception as e:
            logging.error(f"Error loading exercises: {e}")
            raise

    def _create_exercise_embedding(self, exercise: Exercise) -> str:
        """Create a string representation of exercise for embedding"""
        return f"""
        Exercise Name: {exercise.name}
        Category: {exercise.category}
        Difficulty: {exercise.difficulty}
        Intensity: {exercise.intensity}
        Suitable for: {', '.join(exercise.suitable_conditions)}
        Not suitable for: {', '.join(exercise.contraindications)}
        Description: {exercise.description}
        Benefits: {', '.join(exercise.benefits)}
        """

    async def create_embeddings(self) -> None:
        """Create embeddings for all exercises"""
        try:
            # Initialize FAISS index
            self.vector_store = faiss.IndexFlatL2(self.dimension)
            
            # Create embeddings for each exercise
            embeddings = []
            for exercise in self.exercises:
                exercise_text = self._create_exercise_embedding(exercise)
                
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=exercise_text
                )
                
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            
            # Convert to numpy array and add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.vector_store.add(embeddings_array)
            
            # Save to disk
            self.save_index()
            
            logging.info(f"Created and saved embeddings for {len(self.exercises)} exercises")
            
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            raise

    def search_exercises(self, query: str, num_results: int = 2) -> List[Exercise]:
        """Search for exercises based on query"""
        try:
            # Create embedding for query
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.vector_store.search(query_embedding, num_results)
            
            # Return matched exercises
            return [self.exercises[idx] for idx in indices[0]]
            
        except Exception as e:
            logging.error(f"Error searching exercises: {e}")
            return []

# Example usage
async def main():
    db = ExerciseVectorDB()
    
    # Try to load existing index
    if not db.load_index():
        # If no existing index, create new one
        logging.info("Creating new index...")
        db.load_exercises('old_adults RAG/exercises/exercise.json')
        await db.create_embeddings()
    
    # Example search
    user_query = "I have mild arthritis and want to improve my balance. I prefer low-intensity exercises."
    matched_exercises = await db.search_exercises(user_query)
    
    # Print results
    for exercise in matched_exercises:
        print(f"\nName: {exercise.name}")
        print(f"Difficulty: {exercise.difficulty}")
        print(f"Description: {exercise.description}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())