#!/usr/bin/env python3
# License:  Apache 2.0


import argparse
import typing
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertForSequenceClassification,
                          Trainer, TrainingArguments)

TASK2LABELS = {
    "pn": ["neu", "pos", "neg"],
    "rhr": ["no", "yes"],
    "rte": ["NE", "E"],
}
TASK2COLUMNNUM = {
    "pn": 5,
    "rhr": 5,
    "rte": 7,
}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_data(path_data_list: typing.List[Path], target: str, task: str):
    texts_a: typing.List[str] = []
    if task == 'rte':
        texts_b: typing.Optional[typing.List[str]] = []
    else:
        texts_b = None
    labels = []
    origlabel2label = [0, 1, 2]
    for mypath in path_data_list:
        mypath = mypath.expanduser()
        for path in mypath.parent.glob(mypath.name):
            with path.open() as inf:
                for line in inf:
                    items = line[:-1].split("\t")
                    assert len(items) == TASK2COLUMNNUM[task]
                    if items[-1] != target:
                        continue
                    if task == 'rte':
                        assert texts_b is not None
                        texts_a.append(items[2])
                        texts_b.append(items[3])
                    else:
                        texts_a.append(items[2])
                    labels.append(origlabel2label[int(items[1])])

    if len(labels) == 0:
        raise KeyError("No examples are given or invalid path")
    return texts_a, texts_b, labels


def evaluate(*,
             path_model: Path, path_data_list: typing.List[Path],
             path_output: Path,
             task: str,
             max_length: int,
             ):
    label_set: typing.List[str] = TASK2LABELS[task]

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(path_model).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path_model)
    test_texts_a, test_texts_b, test_labels \
        = read_data(path_data_list, 'test', task)

    with path_output.open('w') as outf:
        if test_texts_b is None:
            test_texts_b = [None] * len(test_texts_a)

        for (text_a, text_b, gold_label_idx) in zip(test_texts_a, test_texts_b, test_labels):
            if text_b is None:
                source = tokenizer.batch_encode_plus([text_a],
                                                     padding=True,
                                                     return_tensors='pt')
            else:
                source = tokenizer.batch_encode_plus([[text_a, text_b]],
                                                     padding=True,
                                                     return_tensors='pt')
            source.to(device)
            outputs = model(
                input_ids=source['input_ids'],
                token_type_ids=source['token_type_ids'],
                attention_mask=source['attention_mask'],
            )
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            sys_label = label_set[np.argmax(predictions[0])]
            gold_label = label_set[gold_label_idx]

            outf.write(f'{gold_label}\t{sys_label}\t{predictions[0]}\t{text_a}')
            if text_b is not None:
                outf.write(f'\t{text_b}')
            outf.write('\n')


def main(*, path_data_list: typing.List[Path], path_out: Path, base: str,
         num_train_epochs: int,
         path_log: Path, batch_size: int, log_step: int,
         warmup_step: int, weight_decay: float, max_length: int, task: str,
         ):
    tokenizer = AutoTokenizer.from_pretrained(base)

    train_texts_a, train_texts_b, train_labels \
        = read_data(path_data_list, 'train', task)
    train_encodings = tokenizer(text=train_texts_a, text_pair=train_texts_b,
                                truncation=True,
                                padding=True, max_length=max_length)
    train_dataset = MyDataset(train_encodings, train_labels)

    eval_texts_a, eval_texts_b, eval_labels \
        = read_data(path_data_list, 'dev', task)
    eval_encodings = tokenizer(text=eval_texts_a, text_pair=eval_texts_b,
                               truncation=True,
                               padding=True, max_length=max_length)
    eval_dataset = MyDataset(eval_encodings, eval_labels)

    training_args = TrainingArguments(
        output_dir=path_out,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_step,
        weight_decay=weight_decay,
        logging_dir=path_log,
        logging_steps=log_step,
    )

    label_map: typing.Dict[int, str] \
        = {i: label for i, label in enumerate(TASK2LABELS[task])}

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=base,
        num_labels=len(TASK2LABELS[task]),
        id2label=label_map,
        label2id={label: i for i, label in enumerate(TASK2LABELS[task])},
    )
    model = BertForSequenceClassification.from_pretrained(base,
                                                          config=config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


def get_opts() -> argparse.Namespace:
    default_base: str = "cl-tohoku/bert-base-japanese-v2"
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--inputs", "-i", action="append",
                         required=True, type=Path)
    oparser.add_argument("--output", "-o", required=True, type=Path)
    oparser.add_argument("--task", choices=["pn", "rhr", "rte"], required=True)
    oparser.add_argument("--base",
                         default=default_base)
    oparser.add_argument("--log", default="logs", type=Path)
    oparser.add_argument("--batch", type=int, default=128)
    oparser.add_argument("--log_step", type=int, default=10)
    oparser.add_argument("--epoch", type=int, default=3)
    oparser.add_argument("--warmup_step", type=int, default=500)
    oparser.add_argument("--weight_decay", type=float, default=0.01)
    oparser.add_argument("--max_length", type=int, default=32)

    oparser.add_argument("--evaluate", action="store_true")
    return oparser.parse_args()


if __name__ == '__main__':
    opts = get_opts()
    if opts.evaluate:
        evaluate(
            path_model=Path(opts.base),
            path_data_list=opts.inputs,
            path_output=opts.output,
            task=opts.task,
            max_length=opts.max_length,
        )
    else:
        main(path_data_list=opts.inputs, path_out=opts.output,
             base=opts.base, num_train_epochs=opts.epoch,
             path_log=opts.log, batch_size=opts.batch, log_step=opts.log_step,
             warmup_step=opts.warmup_step, weight_decay=opts.weight_decay,
             max_length=opts.max_length, task=opts.task)
