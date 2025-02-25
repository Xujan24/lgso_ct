"""
Script to generate preference dataset in the conversational format.
The input datafile must be a .tsv file separated by a tabspace and the expected column titles as ['term', 'relation', 'value']
The generated data will be of the format:
[
    {'chosen': [
        {
            'role': 'user', 
            'content': <query>
        }, {
            'role': 'agent', 
            'content': <winning option>
        }
    ],
    'rejected': [
        {
            'role': 'user', 
            'content': <query>
        }, {
            'role': 'agent', 
            'content': <losing option>
        }
    ]},
    ...
]

The 'value' column is used as the winning options and model response as the losing option.
Each model response is paired with all wining options.

**Note: if a term-relation pair has multiple values (e.g. a term having multiple synonyms), then separate each value with a semiclon (;).**
"""


import os
import argparse
from typing import List
from itertools import product
import pandas as pd
import numpy as np
from utils.helpers import gen_query, save_to_pkl, get_responses_from_ref_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def split_df_into_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    n_batches = int(np.ceil(len(df) / batch_size))
    indices = np.array_split(df.index, n_batches)
    batches = [df.loc[batch_indices] for batch_indices in indices]

    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help='Path to the data file. Accepted data formate is .tsv.')
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path name.")
    parser.add_argument("--batch-size", type=int, default=4, help='Batch size for generating responses.')
    parser.add_argument("--model-id", type=str, required=True, help='Reference model id; str if using Huggingface models or path like if loading model from local folder.')
    parser.add_argument("--max-iter", type=int, default=10, help='Maximum number of iterations, for each query, to generate responses using the reference model.')
    parser.add_argument("--max-new-tokens", type=int, default=256, help='Maximum number of new tokens to generate by the reference model.')
    parser.add_argument("--output", type=str, required=True, help='Path to the output file. The output file will be a pickle file.')

    args = parser.parse_args()

    _, ext = os.path.splitext(args.dataset)
    if ext != '.tsv':
        raise ValueError("The dataset file must be a .tsv file.")


    df = pd.read_csv(args.dataset, sep='\t')[:20]

    batches = split_df_into_batches(df, args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                             device_map=device,  # Use available GPUs or CPU
                                             torch_dtype=torch.bfloat16, # For better performance, use bfloat16 if supported
                                             )
    model.eval()

    data_obj = []
    for i in tqdm(range(len(batches))):
        current_batch = batches[i]

        ## use the ground truth values as the winning choices
        ## some have multiple values separated by ;.
        chosen_lists = [x.split(';') for x in current_batch['value'].tolist()]

        ## generate queries using the term and the relationship identifier
        queries = [gen_query(row['term'], row['relation']) for _, row in current_batch.iterrows()]


        ## generate rejected choices
        ## in our current setting we treat the response generated by the reference model as the rejected choices

        rejected_lists = get_responses_from_ref_model(
            model = model,
            tokenizer = tokenizer,
            query = queries,
            max_iter=args.max_iter,
            max_new_tokens=args.max_new_tokens
            )

        for j in range(len(current_batch)):
            query = queries[j]
            chosen_list = list(set(chosen_lists[j]))
            rejected_list = list(set(rejected_lists[j]))

            pairs = list(product(chosen_list, rejected_list))

            for x, y in pairs:
                if x == y:
                    continue
                
                data_obj.append({
                    'chosen': [
                        {
                            'role': 'user',
                            'content': query
                        },
                        {
                            'role': 'assistant',
                            'content': x
                        }
                    ],
                    'rejected': [
                        {
                            'role': 'user',
                            'content': query
                        },
                        {
                            'role': 'assistant',
                            'content': y
                        }
                    ]
                })
                del x, y
            del chosen_list, rejected_list, pairs

            ## save the checkpoints after processing each batch

            checkpoint = {
                'dataset': args.dataset,
                'batch_size': args.batch_size,
                'model_id': args.model_id,
                'max_iter': args.max_iter,
                'max_new_tokens': args.max_new_tokens,
                'current_batch': i,
                'data_obj': data_obj
            }

            save_to_pkl(args.checkpoint, checkpoint)
        
        del chosen_lists, rejected_lists, queries, current_batch, checkpoint

    save_to_pkl(args.output, data_obj)