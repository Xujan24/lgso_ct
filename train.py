import argparse
import warnings
from utils.trainer import TrainingArguments, Trainer
from utils.helpers import load_from_pkl
from datasets import Dataset
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to training dataset (.pkl file)')
    parser.add_argument('--model-id', type=str, required=True, help='Model id')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--method', type=str, default='dpo')
    parser.add_argument('--train-epochs', type=int, default=20)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--checkpoint-save-steps', type=int, default=20)
    parser.add_argument('--num-save-checkpoints', type=int, default=3)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            warnings.warn('Couldn\'t find any cuda devices; using cpu instead.')
    else:
        device = torch.device('cpu')

    train_args = TrainingArguments(
        output_dir=args.output,
        beta=args.beta,
        learning_rate=args.learning_rate,
        train_epochs=args.train_epochs,
        logging_steps=args.logging_steps,
        checkpoint_save_steps=args.checkpoint_save_steps,
        num_save_checkpoints=args.num_save_checkpoints,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        method=args.method
    )

    train_data = Dataset.from_list(load_from_pkl(args.input)[:20])

    trainer = Trainer(
        model_id=args.model_id,
        device=device,
        train_dataset=train_data,
        args=train_args
    )

    trainer.train(resume=args.resume)

    
