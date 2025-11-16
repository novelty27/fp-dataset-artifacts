import torch, random, numpy as np
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
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
    dataset: str = field(default=None)
    max_length: int = field(default=128)
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)
    hypothesis_only: bool = field(default=False)
    dataset_weights: str = field(default=None)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        per_example_loss = loss_function(logits.view(-1, self.model.config.num_labels),
                                        labels.view(-1))

        # apply weights
        if weights is not None:
            weights = weights.to(per_example_loss.device).view(-1)
            loss = (per_example_loss * weights).sum() / weights.sum()
        else:
            loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss
    
def main():
    argp = HfArgumentParser((TrainingArguments, CustomArgs))

    # Update output_dir to follow this format [current date/time - output_dir - random_seed]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args, args = argp.parse_args_into_dataclasses()
    output_dir = current_time + '-' + training_args.output_dir + '-' + str(training_args.seed)
    training_args.output_dir = output_dir

    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        # default_datasets = {'qa': ('squad',), 'nli': ('snli',), 'mnli':('glue','multi_nli')}
        # dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
        #     default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        dataset_id = 'multi_nli' if args.dataset == 'mnli' else args.dataset
        eval_split = 'validation_matched' if dataset_id == 'multi_nli' else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(dataset_id)

    dataset_weights = None
    if args.dataset_weights is not None:
        with open(args.dataset_weights, "r") as f:
            dataset_weights = json.load(f)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length, args.hypothesis_only, weights=dataset_weights)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    if args.dataset_weights is not None:
        trainer_class = WeightedTrainer
    else:
        trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
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
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="evancarr-university-of-texas-at-austin",
        # Set the wandb project where this run will be logged.
        project="AI388-NLP - Dateset Artifacts",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "google/electra-small-discriminator",
            "dataset": dataset_id,
            "epochs": training_args.num_train_epochs if training_args.do_train else 0,
            "seed": training_args.seed
        },
    )

    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)
        run.log(results)
        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
