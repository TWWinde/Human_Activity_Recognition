import logging
logging.basicConfig(level=logging.INFO)
from train import Trainer
from models.model import LSTM, GRU
from input_pipeline.dataloader import HumanActivityDataset,get_dataloaders, HumanActivityDataset_json
from configs.config import get_config
import torch
from torch.utils.data import random_split, Dataset, DataLoader

def main():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("MPS Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("Device: {}".format(device))
    config = get_config()
    # setup pipeline
    #dataset = HumanActivityDataset(data_dir = config.data_root_dir, seq_length=config.window_size, stride=config.step_size)
    #train_loader, test_loader = get_dataloaders(data_dir=config.data_root_dir, batch_size=config.batch_size, seq_length=config.window_size, stride=config.step_size)

    dataset = HumanActivityDataset_json(data_dir="/Users/tangwenwu/Documents/motion_data.json", seq_length=64, stride=32, mode='train')
    total_size = len(dataset)
    train_size = int(0.9 * total_size)  # 80% 做训练
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 分别创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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

