import torch, random, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, HfArgumentParser
import datasets
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import wandb
from datetime import datetime
from dataclasses import dataclass, field


NUM_PREPROCESSING_WORKERS = 2

@dataclass
class CustomArgs:
    model: str = field(default='google/electra-small-discriminator')
    task: str = field(default='nli')
    dataset: str = field(default='snli')
    max_length: int = field(default=128)
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)
    eval_on_train: bool = field(default=True)
    #resume_from_checkpoint

def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Pull in the command line arguments
    argp = HfArgumentParser((TrainingArguments, CustomArgs))
    training_args, args = argp.parse_args_into_dataclasses()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args.output_dir = current_time + '-' + training_args.output_dir + '-' + str(training_args.seed)

    # Set the random seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    # Prepare the model
    task_kwargs = {'num_labels': 3}
    model_class = AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(training_args.resume_from_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(training_args.resume_from_checkpoint)

    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    # Prepare the dataset 
    if not args.eval_on_train:
        print("This file is used to generate weights on a training set. Pass in `--eval_on_train True")
        return 
    
    dataset_id = args.dataset
    eval_split = 'train' if args.eval_on_train else 'validitaion'
    dataset = datasets.load_dataset(dataset_id)
    
    prepared_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length, hypothesis_only=True)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)


    eval_dataset = dataset[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    eval_dataset_featurized = eval_dataset.map(
        prepared_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    compute_metrics = compute_accuracy
    
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

        # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    results = trainer.evaluate(**eval_kwargs)

    print('Evaluation results:')
    print(results)

    os.makedirs(training_args.output_dir, exist_ok=True)

    with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
        json.dump(results, f)

    with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
        for i, example in enumerate(eval_dataset):
            example_with_prediction = dict(example)
            example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
            example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
            f.write(json.dumps(example_with_prediction))
            f.write('\n')

    # Generate the weights and store them in a file. 
    # Need to know what value to key the weights to os dataset.map() can pull them
    # pairID, weight

if __name__ == "__main__":
    main()
