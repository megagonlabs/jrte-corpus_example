
# Example codes for Japanese Realistic Textual Entailment Corpus

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![ci](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/ci.yml/badge.svg)](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/ci.yml)
[![markdownlint](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/lint.yml/badge.svg)](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/lint.yml)
[![Typos](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/typos.yml/badge.svg)](https://github.com/megagonlabs/jrte-corpus_example/actions/workflows/typos.yml)

- [train.py](train.py) is an example code to exploit [Japanese Realistic Textual Entailment Corpus](https://github.com/megagonlabs/jrte-corpus).
- [ブログ記事: じゃらんnetに投稿された宿クチコミを用いた感情極性分析・含意関係認識の一例](https://www.megagon.ai/jp/blog/japanese-realistic-textual-entailment-corpus/)

## Setup

```console
$ git clone https://github.com/megagonlabs/jrte-corpus
$ poetry install --no-root
```

## Training

```console
$ poetry run python3 ./train.py -i ./jrte-corpus/data/pn.tsv -o ./model-pn --task pn
$ poetry run python3 ./train.py -i ./jrte-corpus/data/rhr.tsv -o ./model-rhr --task rhr
$ poetry run python3 ./train.py -i './jrte-corpus/data/rte.*.tsv' -o ./model-rte --task rte
```

## Serving

```console
$ poetry run transformers-cli serve --task sentiment-analysis --model ./model-pn --port 8900
$ curl -X POST -H "Content-Type: application/json" "http://localhost:8900/forward" -d '{"inputs":["ご飯が美味しいです。", "3人で行きました。" , "部屋は狭かったです。"] }'
{"output":[{"label":"pos","score":0.8015708923339844},{"label":"neu","score":0.47732535004615784},{"label":"neg","score":0.42699119448661804}]}

$ poetry run transformers-cli serve --task sentiment-analysis --model ./model-rhr --port 8901
$ curl -X POST -H "Content-Type: application/json" "http://localhost:8901/forward" -d '{"inputs":["ご飯が美味しいです。", "3人で行きました。"] }'
{"output":[{"label":"yes","score":0.9653761386871338},{"label":"no","score":0.8748807907104492}]}

$ poetry run transformers-cli serve --task sentiment-analysis --model ./model-rte --port 8902
$ curl -X POST -H "Content-Type: application/json" "http://localhost:8902/forward" -d '{"inputs":[["風呂がきれいです。", "食事が美味しいです" ] , [ "暑いです。", "とても暑かった"]] }'
{"output":[{"label":"NE","score":0.9982748627662659},{"label":"E","score":0.9790723323822021}]
```

## Evaluation

```console
$ poetry run python3 ./train.py --evaluate -i ./jrte-corpus/data/pn.tsv --base ./model-pn --task pn -o ./model-pn/evaluate_output.txt
$ awk '{if($1==$2){ok+=1} } END{ print(ok, NR, ok/NR) }' ./model-pn/evaluate_output.txt
464 553 0.83906

$ poetry run python3 ./train.py --evaluate -i ./jrte-corpus/data/rhr.tsv --base ./model-rhr --task rhr -o ./model-rhr/evaluate_output.txt
$ awk '{if($1==$2){ok+=1} } END{ print(ok, NR, ok/NR) } ' ./model-rhr/evaluate_output.txt
490 553 0.886076

$ poetry run python3 ./train.py --evaluate -i './jrte-corpus/data/rte.*.tsv' --base ./model-rte --task rte -o ./model-rte/evaluate_output.txt
$ awk '{if($1==$2){ok+=1} } END{ print(ok, NR, ok/NR) } ' ./model-rte/evaluate_output.txt
4903 5529 0.886779
```

## Prediction

```console
$ echo -e '飯が美味しいです。\n3人で行きました。\n部屋は狭かったです。' | poetry run python3 ./train.py --predict --base ./model-pn --task pn
pos	[0.02000473439693451, 0.9655841588973999, 0.014411096461117268]
neu	[0.6542302370071411, 0.3007880747318268, 0.04498171806335449]
neg	[0.12337885051965714, 0.1064520850777626, 0.7701690196990967]

$ echo -e 'ご飯が美味しいです。\n3人で行きました。' | poetry run python3 ./train.py --predict --base ./model-rhr --task rhr
yes	[0.019858278334140778, 0.9801417589187622]
no	[0.9749531149864197, 0.02504686638712883]

$  echo -e '風呂がきれいです。\t食事が美味しいです\n暑いです。\tとても暑かった' | poetry run python3 ./train.py --predict --base ./model-rte --task rte
NE	[0.9982748031616211, 0.0017251790268346667]
E	[0.02092761918902397, 0.9790723919868469]
```
