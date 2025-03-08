# Description: 配置文件，用于存放模型的参数配置
import os
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="HAR Model Configuration")

    # 主要参数
    parser.add_argument('--data_root_dir', type=str, default="/Users/tangwenwu/Downloads/HAPT Data Set")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--step_size', type=int, default=64)
    parser.add_argument('--transform', type=bool, default=True)
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru'])

    parser.add_argument("--input_size", type=int, default=6, help="Feature input size")
    parser.add_argument("--window_length", type=int, default=128, help="Input sequence length")
    parser.add_argument("--rnn_units", type=int, default=128, help="Number of LSTM units")
    parser.add_argument("--rnn_num", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--rnn_dropout", type=float, default=0.3, help="Dropout rate for LSTM")
    parser.add_argument("--dense_units", type=int, default=64, help="Dense layer units")
    parser.add_argument("--dense_num", type=int, default=1, help="Number of dense layers")
    parser.add_argument("--dense_dropout", type=float, default=0.3, help="Dropout rate for dense layers")
    parser.add_argument("--n_classes", type=int, default=12, help="Number of output classes")

    # 训练参数
    parser.add_argument('--total_steps', type=int, default=1500)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--ckpt_interval', type=int, default=500)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--acc_weight', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--checkpoint_paths', type=str, default="/Users/tangwenwu/Documents/GitHub/Human_activity_recognition/human_activity_recognition/output/checkpoint")
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])

    args = parser.parse_args()
    return args

# 使用示例
if __name__ == "__main__":
    config = get_config()
    print(config)