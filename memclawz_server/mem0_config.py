"""
Mem0 Configuration for MemClawz v2.0
"""
import os
from mem0 import Memory

# Try different vector store configurations
def create_mem0_memory():
    """Create Mem0 memory instance with fallback configurations"""
    
    # Configuration attempts in order of preference:
    configs = [
        # Try Qdrant (if running)
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memclawz_v2",
                    "host": "localhost",
                    "port": 6333,
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        },
        # Fallback: Local file-based storage with local embeddings
        {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "memclawz_v2",
                    "path": os.path.expanduser("~/.openclaw/mem0-storage")
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        },
        # Final fallback: In-memory with local embeddings
        {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "memclawz_v2",
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        }
    ]
    
    for config in configs:
        try:
            print(f"Trying Mem0 config: {config['vector_store']['provider']}")
            memory = Memory.from_config(config)
            print(f"✓ Mem0 initialized with {config['vector_store']['provider']}")
            return memory
        except Exception as e:
            print(f"✗ Failed with {config['vector_store']['provider']}: {e}")
            continue
    
    # Ultimate fallback: basic memory
    try:
        print("Trying basic Mem0 configuration...")
        memory = Memory()
        print("✓ Mem0 initialized with default configuration")
        return memory
    except Exception as e:
        print(f"✗ Failed with default configuration: {e}")
        raise RuntimeError("Could not initialize Mem0 with any configuration")

if __name__ == "__main__":
    # Test configuration
    try:
        mem = create_mem0_memory()
        print("Mem0 configuration successful!")
        
        # Test basic functionality
        test_text = "This is a test memory for MemClawz v2.0"
        result = mem.add(test_text, user_id="test")
        print(f"Test add result: {result}")
        
        search_result = mem.search(test_text, user_id="test")
        print(f"Test search result: {search_result}")
        
    except Exception as e:
        print(f"Configuration test failed: {e}")