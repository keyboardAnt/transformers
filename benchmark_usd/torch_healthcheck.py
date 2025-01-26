import torch

def main():
    print("\nPyTorch CUDA Health Check:")
    print("-" * 30)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Additional memory info
        print("\nGPU Memory Info:")
        print("-" * 30)
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    main() 