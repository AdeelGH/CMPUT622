# CMPUT622
Final Project for Trustworthy ML

## Execution

## Baseline Execution
Example for execution without perturbations/collocations of any kind with the RTE dataset:
```
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name rte \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir /tmp/rte/ \
  --overwrite_output_dir \
  --seed 124 \
```


## Perturbed Execution
Use the `run_glue.sh` script to run the perturbations.

## References
- https://github.com/sjmeis/CLMLDP
- https://github.com/sjmeis/MLDP
- https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
- https://www.youtube.com/watch?v=fDzCBKArdYg
- https://huggingface.co/docs/transformers/installation#install-from-source
- https://github.com/protocolbuffers/protobuf/tree/main/python#installation

## Notes
To run CollocationExtractor and DatasetPerturbation you will need to have a local copy of their models and also the data.zip file that contains bigrams and trigrams from their repository. Not included in this repo since they're large files.

