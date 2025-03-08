import logging
logging.basicConfig(level=logging.INFO)
from train import Trainer
from models.model import LSTM, GRU
from input_pipeline.dataloader import HumanActivityDataset,get_dataloaders
from configs.config import get_config
import torch
from torch.utils.data import Dataset, DataLoader

def main():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("MPS Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("Device", device)
    config = get_config()
    # setup pipeline
    dataset = HumanActivityDataset(data_dir = config.data_root_dir, seq_length=config.window_size, stride=config.step_size)
    train_loader, test_loader = get_dataloaders(data_dir=config.data_root_dir, batch_size=config.batch_size, seq_length=config.window_size, stride=config.step_size)

    logging.info("Data loaded")

    if config.model == 'lstm':
        model = LSTM(
        input_size=config.input_size,
        n_classes=config.n_classes,
        window_length=config.window_length,
        rnn_units=config.rnn_units,
        rnn_num=config.rnn_num,
        rnn_dropout=config.rnn_dropout,
        dense_units=config.dense_units,
        dense_num=config.dense_num,
        dense_dropout=config.dense_dropout
    ).to(device)
    elif config.model == 'gru':
        model = GRU(config).to(device)
    else:
        print('Error, no model is fund')

    print(model)

    
    trainer = Trainer(config, model, train_loader, test_loader)
    trainer.train()

   

if __name__ == "__main__":
    main()

