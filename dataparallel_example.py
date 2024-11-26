import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import time
from tqdm import tqdm

class InferenceDataset(Dataset):
    def __init__(self, data: List):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def inference_dp(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None
) -> List:
    """
    Run inference using DataParallel
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Wrap model in DataParallel if multiple GPUs available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # Scale batch size by number of GPUs
    effective_batch_size = batch_size * torch.cuda.device_count() if device == 'cuda' else batch_size
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Run inference
    predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, (tuple, list)):
                batch = [b.to(device) for b in batch]
            else:
                batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            predictions.extend(outputs.cpu())
            
            # Optional: Clear cache periodically
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    return predictions

def main():
    # Example usage
    model = YourModel()  # Define your model
    data = [...]  # Your data
    dataset = InferenceDataset(data)
    
    predictions = inference_dp(
        model=model,
        dataset=dataset,
        batch_size=32,
        num_workers=4
    )
    
    # Process predictions
    for pred in predictions:
        # Your post-processing logic
        pass

if __name__ == "__main__":
    main()