"""
Trainer class to finetune mode using DPO and LPO.
** Note: For DPO training the huggingface implementation could provide more efficient implementation (https://huggingface.co/docs/trl/main/en/dpo_trainer) **
"""
import time
import os
import shutil
import math
import warnings
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
from .helpers import save_to_pkl


class TrainingArguments():
    def __init__(
            self,  
            output_dir: str,
            checkpoint_dir: str, 
            beta: float = 0.1, 
            learning_rate: float = 1e-6, 
            train_epochs: int=10, 
            logging_steps: int = 10,
            checkpoint_save_steps: int = 50,
            num_save_checkpoints: int = 3, 
            batch_size: int = 32, 
            label_smoothing: float = 0.0,
            method: str = 'dpo' ## lpo if ranked dataset
        ) -> None:
        self.beta = beta
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.logging_steps = logging_steps
        self.checkpoint_save_steps = checkpoint_save_steps
        self.num_save_checkpoints = num_save_checkpoints
        self.label_smoothing = label_smoothing
        self.method = method


class Trainer():
    def __init__(self, model_id: str, args: TrainingArguments, train_dataset: Dataset, device: torch.device) -> None:
        self.model_id = model_id
        self.args = args
        self.train_dataset = train_dataset
        self.device = device

        ## create model and tokenizer
        self.__create_model_and_tokenizer()

        ## create reference model
        self.__create_ref_model()

        ## placeholder to collect the chosen and rejected rewards during training
        self.chosen_rewards_per_batch_per_epoch = []
        self.rejected_rewards_per_batch_per_epoch = []

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)

        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
    

    def train(self, resume: bool = False) -> None:
        batches = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)

        current_epoch = 0
        current_step = 0
        current_batch_idx = 0
        total_steps = len(batches) * self.args.train_epochs

        if resume:
            checkpoints = os.listdir(self.args.checkpoint_dir)
            if len(checkpoints) > 0:
                tqdm.write('Loading latest checkpoint...')
                dir = self.__get_dir(checkpoints, latest=True)

                ## load the checkpoint and load the model and optimizer from it.
                checkpoint = torch.load(os.path.join(self.args.checkpoint_dir, dir, 'checkpoint.pt'))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                ## load training steps from the checkpoint
                current_step = checkpoint['step']
                current_epoch = math.floor(current_step / len(batches))
                current_batch_idx = current_step - current_epoch * len(batches)

                ## load chosen and rejected rewards from the checkpoint
                self.chosen_rewards_per_batch_per_epoch = checkpoint['chosen_rewards']
                self.rejected_rewards_per_batch_per_epoch = checkpoint['rejected_rewards']
            else:
                warnings.warn('No checkpoints found. Training from scratch.')
        else:
            ## remove previous checkpoints if exists and not resuming
            tqdm.write('Removing previous checkpoints if exists...')
            for dir in os.listdir(self.args.checkpoint_dir):
                shutil.rmtree(os.path.join(self.args.checkpoint_dir, dir))

        progressbar = tqdm(total=total_steps, position=0, initial=current_step, leave=True)

        for epoch in range(current_epoch, self.args.train_epochs):
            running_loss = 0.0

            ## if resuming then load from the previous checkpoints
            running_chosen_rewards_per_batch = self.chosen_rewards_per_batch_per_epoch[-1] if resume and epoch == current_epoch else []
            running_rejected_rewards_per_batch = self.rejected_rewards_per_batch_per_epoch[-1] if resume and epoch == current_epoch else []
            # self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(batches):
                if epoch == current_epoch and batch_idx < current_batch_idx:
                    continue
                ## the LPO dataset will have lambda scores along with query, chosen and rejected options.
                if self.args.method == 'lpo':
                    if 'lambda' not in list(batch.keys()):
                        raise Exception('LPO dataset must have lambda scores')
                    else:
                        lambdas = batch['lambda'].to(self.device)

                batch = self.__gen_tokenized_inputs(batch)

                ## for efficiency both the chosen and rejected inputs are concatenated
                ## so that the logits can be computed in a single forward pass
                ## input_ids.shape = (2*batch_size, seq_len, vocab_size)
                ## the first half are the chosen inputs and the bottom half are the rejected inputs
                chosen_input_ids, rejected_input_ids = torch.chunk(batch['input_ids'], 2, dim=0)
                chosen_attention_mask, rejected_attention_mask = torch.chunk(batch['attention_mask'], 2, dim=0)

                response_logits = self.model(**batch).logits

                with torch.no_grad():
                    ref_response_logits = self.ref_model(**batch).logits
                
                q_chosen_logits, q_rejected_logits = torch.chunk(response_logits, 2, dim=0)
                ref_q_chosen_logits, ref_q_rejected_logits = torch.chunk(ref_response_logits, 2, dim=0)
                
                ## get the logits for the response tokens only
                chosen_logits = self.compute_logits(
                    logits=q_chosen_logits, 
                    input_ids=chosen_input_ids, 
                    attention_mask=chosen_attention_mask, 
                    query_token_ids=batch['q_input_ids']
                )

                rejected_logits = self.compute_logits(
                    logits=q_rejected_logits, 
                    input_ids=rejected_input_ids, 
                    attention_mask=rejected_attention_mask, 
                    query_token_ids=batch['q_input_ids']
                )

                ref_chosen_logits = self.compute_logits(
                    logits=ref_q_chosen_logits, 
                    input_ids=chosen_input_ids, 
                    attention_mask=chosen_attention_mask, 
                    query_token_ids=batch['q_input_ids']
                )

                ref_rejected_logits = self.compute_logits(
                    logits=ref_q_rejected_logits, 
                    input_ids=rejected_input_ids, 
                    attention_mask=rejected_attention_mask, 
                    query_token_ids=batch['q_input_ids']
                )
                
                ## compute loss
                if self.args.method == 'dpo':
                    losses, chosen_rewards, rejected_rewards = self.compute_dpo_loss(
                        chosen_logits=chosen_logits,
                        rejected_logits=rejected_logits,
                        ref_chosen_logits=ref_chosen_logits,
                        ref_rejected_logits=ref_rejected_logits,
                    )

                elif self.args.method == 'lpo':
                    losses, chosen_rewards, rejected_rewards = self.compute_lpo_loss(
                        chosen_logits=chosen_logits,
                        rejected_logits=rejected_logits,
                        ref_chosen_logits=ref_chosen_logits,
                        ref_rejected_logits=ref_rejected_logits,
                        lambdas=lambdas
                    )
                
                ## calc the average of the losses and back propagate the loss
                loss = losses.mean()
                loss.backward()

                self.optimizer.step()

                current_step += 1
                running_loss += loss.item()
                running_chosen_rewards_per_batch.append(chosen_rewards.mean().item())
                running_rejected_rewards_per_batch.append(rejected_rewards.mean().item())

                ## print summary
                if current_step % self.args.logging_steps == 0:
                    tqdm.write(f"[epoch: {round(current_step/len(batches), 2)}, loss: {round(running_loss/self.args.logging_steps, 4)}]")
                    running_loss = 0

                ## save checkpoint
                if current_step % self.args.checkpoint_save_steps == 0:
                    tqdm.write(f'Saving checkpoint at step {current_step}...')
                    self.__save_checkpoint(current_step)

                progressbar.update(1)

            ## save the avg. chosen and rejected rewards per batch for the epoch
            self.chosen_rewards_per_batch_per_epoch.append(running_chosen_rewards_per_batch)
            self.rejected_rewards_per_batch_per_epoch.append(running_rejected_rewards_per_batch)
        
        chosen_rejected_rewards = {
            'chosen_rewards': self.chosen_rewards_per_batch_per_epoch,
            'rejected_rewards': self.rejected_rewards_per_batch_per_epoch
        }

        ## save the reward to a file
        save_to_pkl(file_path=f'{self.args.output_dir}/rewards/{self.model_id}.pkl', data=chosen_rejected_rewards)

        ## save the fine-tuned model
        self.__save()


    def compute_logits(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        query_token_ids: torch.tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = logits.shape ## B x L x V

        ## get response start indices
        response_starts = torch.tensor([len(q_tokens) for q_tokens in query_token_ids], device=self.device) ## B x 1

        ## generate index for batchwise indexing
        indices = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) ## B x L

        ## mask response tokens (exclude the query and padding tokens) and get response token ids
        response_mask = (indices >= response_starts[:, None]) & (attention_mask.bool()) ## B x L
        response_ids = input_ids.masked_fill(~response_mask, 0) ## B x L

        ## compute log probabilities
        ## first compute the log probabilities for each tokens
        log_probs = F.log_softmax(logits, dim=-1) ## B x L x V

        ## get log probabilities for the response tokens only
        response_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1) ## B x L
        response_log_probs = response_log_probs * response_mask

        ## finally sum up the log probabilities and return.
        return response_log_probs.sum(dim=-1) ## B x 1
    

    def compute_lpo_loss(
        self,
        chosen_logits: torch.FloatTensor, 
        rejected_logits: torch.FloatTensor, 
        ref_chosen_logits: torch.FloatTensor, 
        ref_rejected_logits: torch.FloatTensor, 
        lambdas: torch.Tensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    
        pi_logratios = chosen_logits - rejected_logits
        ref_logratios = ref_chosen_logits - ref_rejected_logits

        logits = pi_logratios - ref_logratios

        losses = -lambdas * (F.logsigmoid(self.args.beta * logits) * (1-self.args.label_smoothing) 
                             - F.logsigmoid(self.args.beta * logits) * self.args.label_smoothing)
        
        loss = losses.mean()
        chosen_rewards = self.args.beta * (chosen_logits - ref_chosen_logits)
        rejected_rewards = self.args.beta * (rejected_logits - ref_rejected_logits)

        return loss, chosen_rewards, rejected_rewards
    

    def compute_dpo_loss(
        self,
        chosen_logits: torch.FloatTensor, 
        rejected_logits: torch.FloatTensor, 
        ref_chosen_logits: torch.FloatTensor, 
        ref_rejected_logits: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    
        pi_logratios = chosen_logits - rejected_logits
        ref_logratios = ref_chosen_logits - ref_rejected_logits

        logits = pi_logratios - ref_logratios

        losses = (-F.logsigmoid(self.args.beta * logits) * (1-self.args.label_smoothing) 
                  - F.logsigmoid(self.args.beta * logits) * self.args.label_smoothing)
        
        loss = losses.mean()
        chosen_rewards = self.args.beta * (chosen_logits - ref_chosen_logits).detach()
        rejected_rewards = self.args.beta * (rejected_logits - ref_rejected_logits).detach()

        return loss, chosen_rewards, rejected_rewards
    

    def __create_model_and_tokenizer(self) -> None:
        ## create tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map=self.device)


    def __create_ref_model(self) -> None:
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map=self.device)
        # Freeze the reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False


    def __gen_tokenized_inputs(self, samples):
        formatted_samples = self.__concate_prompt_and_response(samples)

        ## we will concat the choosen inputs and rejected inputs so the we can get the logits with a single pass
        inputs = [*formatted_samples['q_chosen'], *formatted_samples['q_rejected']]

        tokenized_q = self.tokenizer(formatted_samples['query'], add_special_tokens=False, padding=False).to(self.device)
        tokenized_inputs = self.tokenizer(inputs, padding=True, padding_side='right', return_tensors="pt").to(self.device)

        return {
            'q_input_ids': tokenized_q.input_ids,
            'q_attention_mask': tokenized_q.attention_mask,
            'input_ids': tokenized_inputs.input_ids,
            'attention_mask': tokenized_inputs.attention_mask,
        }


    @staticmethod
    def __concate_prompt_and_response(samples):
        ## identify the preference data format
        ## conversation format will have two keys ['choosen', 'rejected']
        ## otherwise it will have four keys ['prompt', 'choosen', 'rejected', 'lambda]
        keys = list(samples.keys())

        if len(keys) == 2:
            q = samples['chosen'][0]['content']
            chosen = samples['chosen'][1]['content']
            rejected = samples['rejected'][1]['content']
            
        elif len(keys) == 4:
            q = samples['query']
            chosen = samples['chosen']
            rejected = samples['rejected']
        
        q_chosen = [f"{x} {y}" for x, y in zip(q, chosen)]
        q_rejected = [f"{x} {y}" for x, y in zip(q, rejected)]

        return {
            'query': q,
            'q_chosen': q_chosen,
            'q_rejected': q_rejected,
        }


    def __save(self) -> None:
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        self.model.save_pretrained(os.path.join(self.args.output_dir, self.model_id.split('/')[-1]))
        self.tokenizer.save_pretrained(os.path.join(self.args.output_dir, self.model_id.split('/')[-1]))
        return None


    def __save_checkpoint(self, current_step: int) -> None:
        """Saves the current model checkpoint."""
        ## check if we want to save the checkpoint or not
        if current_step == 0:
            return None
        
        ## get a list of current saved checkpoints
        current_checkpoint_dir = os.path.join(self.args.checkpoint_dir, "checkpoint_{}".format(current_step))
        previous_checkpoints = os.listdir(self.args.checkpoint_dir)

        ## only store last self.args.num_save_checkpoints checkpoints
        if len(previous_checkpoints) >= self.args.num_save_checkpoints:
            remove_dir = self.__get_dir(previous_checkpoints)
            shutil.rmtree(os.path.join(self.args.checkpoint_dir, remove_dir))
        
        # create the directory for current checkpoint and save the model checkpoints
        os.makedirs(current_checkpoint_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': current_step,
            'chosen_rewards': self.chosen_rewards_per_batch_per_epoch,
            'rejected_rewards': self.rejected_rewards_per_batch_per_epoch

        }
        torch.save(checkpoint, os.path.join(current_checkpoint_dir, "checkpoint.pt"))
    

    def __get_dir(self, dir_list: List[str], latest: bool = False) -> str:
        """Given a list of dirs, returns the oldest (or newest) dir.

        Arguments:
            - dir_list (List[str]) -- a list of directories
            - latest (bool) -- if `True` returns the newest dir, else returns the oldest dir. default `False`

        Return `str` -- the oldest (or newest) dir.
        
        """
        sorted_dirs = sorted(dir_list, key=lambda x: os.path.getctime(os.path.join(self.args.checkpoint_dir, x)))
        return sorted_dirs[-1] if latest else sorted_dirs[0]