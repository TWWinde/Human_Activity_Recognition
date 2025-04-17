from models.model import LSTM, GRU
from input_pipeline.dataloader import HumanActivityDataset,get_dataloaders
from configs.config import get_config
import torch
import coremltools as ct
import logging
import os
import numpy as np


config = get_config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logging.info("Device: {}".format(device))
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

best_model_path = os.path.join(config.checkpoint_paths, "best_model.pth")
last_model_path = os.path.join(config.checkpoint_paths, "final_model.pth")

if os.path.exists(best_model_path):
    checkpoint_path = best_model_path
elif os.path.exists(last_model_path):
    checkpoint_path = last_model_path
else:
    logging.info("No checkpoint found. Starting fresh!")


checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
example_input = torch.randn(1, 64, 6).to(device)  # 匹配训练时的输入形状
traced_model = torch.jit.trace(model, example_input)
os.makedirs(config.ct_model_paths, exist_ok=True)
traced_model.save(os.path.join(config.ct_model_paths, "sensor_model.pt"))


ml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    classifier_config=ct.ClassifierConfig(["Walk", "Run", "Sit" , "Lay", "Jump"])
)

ml_model.save(os.path.join(config.ct_model_paths, "ActivityClassifier.mlpackage"))

ml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    convert_to="neuralnetwork",
    classifier_config=ct.ClassifierConfig(["Walk", "Run", "Sit" , "Lay", "Jump"])
)

ml_model.save(os.path.join(config.ct_model_paths, "ActivityClassifier.mlmodel"))


numpy_input = np.random.rand(1, 64, 6).astype(np.float32)
coreml_pred = ml_model.predict({"input": numpy_input})
print(coreml_pred)
