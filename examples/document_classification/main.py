import json
import logging
import os
from pathlib import Path
from argparse import Namespace

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForDocumentClassification
from .utils import convert_documents_to_features, parse_mldoc


logger = logging.getLogger(__name__)


@click.group(name="document-classification")
def cli():
    pass


@cli.command()
@click.option("--train-data-file", type=click.Path(exists=True))
@click.option("--validation-data-file", type=click.Path(exists=True))
@click.option("-t", "--test-set", multiple=True)
@click.option("--do-train/--no-train", default=False)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--num-train-epochs", default=5)
@click.option("--train-batch-size", default=16)
@click.option("--max-seq-length", default=512)
@click.option("--masked-entity-prob", default=0.7)
@click.option(
    "--context-entity-selection-order", default="highest_prob", type=click.Choice(["natural", "random", "highest_prob"])
)
@click.option("--document-split-mode", default="simple", type=click.Choice(["simple", "per_mention"]))
@click.option("--fix-entity-emb/--update-entity-emb", default=True)
@click.option("--seed", default=1)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    train_data = convert_documents_to_features(parse_mldoc(args.train_data_file), args.tokenizer, args.max_seq_length)
    dataset = {"train": train_data}
    if args.validation_data_file:
        dataset["validation"] = convert_documents_to_features(
            parse_mldoc(args.train_data_file), args.tokenizer, args.max_seq_length
        )

    model_config = args.model_config
    logger.info("Model configuration: %s", model_config)

    model = LukeForDocumentClassification(model_config)
    model.load_state_dict(args.model_weights, strict=False)
    model.to(args.device)

    def collate_fn(batch):
        def create_padded_sequence(attr_name: str, padding_value: int):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
        )

        ret["label"] = torch.LongTensor([o.label for o in batch])

        return ret

    if args.do_train:

        train_dataloader = DataLoader(
            dataset["train"], batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True
        )

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        best_dev_accuracy = [-1]
        best_weights = [None]

        def step_callback(model: LukeForDocumentClassification, global_step: int):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                validation_dataloader = DataLoader(
                    dataset["validation"], batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=False
                )

                dev_results = evaluate(args, validation_dataloader, model)
                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
                tqdm.write("dev: " + str(dev_results))

                if dev_results["accuracy"] > best_dev_accuracy[0]:
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_accuracy[0] = dev_results["f1"]
                    results["best_epoch"] = epoch

                model.train()

        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        trainer.train()

        if args.output_dir:
            logger.info("Saving model to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}

    if args.do_eval:
        model.eval()

        for test_file_path in args.test_set:
            logger.info("***** Evaluating: %s *****", test_file_path)
            eval_data = parse_mldoc(test_file_path)
            eval_data = convert_documents_to_features(eval_data, args.tokenizer, args.max_seq_length,)
            eval_dataloader = DataLoader(
                eval_data, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=False
            )
            predictions_file = None
            if args.output_dir:
                predictions_file = os.path.join(args.output_dir, f"eval_predictions_{Path(test_file_path).name}.jsonl")
            results[test_file_path] = evaluate(args, eval_dataloader, model, predictions_file)

        if args.output_dir:
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

    return results


def evaluate(args, eval_dataloader: DataLoader, model: LukeForDocumentClassification, output_file: str = None):
    predictions = []
    labels = []
    for batch in tqdm(eval_dataloader, leave=False):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            logits = model(**inputs)
            result = torch.argmax(logits, dim=1)
            predictions += result.tolist()
        labels += batch["label"].tolist()

    num_correct = 0
    num_total_data = 0
    eval_predictions = []
    for prediction, label in zip(predictions, labels):
        num_total_data += 1
        if prediction == label:
            num_correct += 1

    if output_file:
        with open(output_file, "w") as f:
            for obj in eval_predictions:
                f.write(json.dumps(obj) + "\n")

    accuracy = num_correct / num_total_data

    logger.info("accuracy: %.5f", accuracy)
    logger.info("#data: %d", num_total_data)
    logger.info("#correct: %d", num_correct)

    return dict(accuracy=accuracy)
