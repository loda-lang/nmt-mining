# NMT Program Mining
LODA Program Mining using [Neural Machine Translation (NMT)](https://github.com/tensorflow/nmt). Contributed by [JUrban](https://github.com/JUrban) and [barakeel](https://github.com/barakeel).

## Create Training Data

The training data is generated from [loda-programs](https://github.com/loda-lang/loda-programs) and stored in the `TRDIR` folder.

```bash
export TRDIR=/traindata
cd loda-programs/oeis
for i in `ls -d */`; do cd $i; for j in *; do perl -pe 's/;.*//; s/^ *//; s/mov/M/; s/add/A/; s/sub/S/; s/trn/T/; s/mul/U/; s/div/D/; s/dif/F/; s/mod/O/; s/pow/P/; s/gcd/G/; s/bin/B/; s/cmp/C/; s/min/I/; s/max/X/; s/lpb/L/; s/lpe/E/; s/clr/R/; s/seq/Q/; s/,/# /g; s/[\$]/\$ /g; s/([0-9])/$1 /g;  s/\-/~ /g;' $j | tr "\n" " " | perl -pe 's/^ *//; s/ +/ /g' > $TRDIR/$i/$j; done; cd ..; done
```

Translate back to LODA:

```
perl -pe 's/M/; mov/g; s/A/; add/g; s/S/; sub/g; s/T/; trn/g; s/U/; mul/g; s/D/; div/g; s/F/; dif/g; s/O/; mod/g; s/P/; pow/g; s/G/; gcd/g; s/B/; bin/g; s/C/; cmp/g; s/I/; min/g; s/X/; max/g; s/L/; lpb/g; s/E/; lpe/g; s/R/; clr/g; s/Q/; seq/g; s/# */,/g; s/[\$] */\$/g; s/([0-9]) */$1/g; s/\~ */-/g; s/^; *//; s/^ *//; s/ *$//; s/ +/ /g' 
```

## Model Training

The folder `TRDIR` contains the training data and `MODDIR` is where the model shall be stored.

```
export CUDA_VISIBLE_DEVICES=0
time python -m nmt.nmt \
  --encoder_type=bi \
  --learning_rate=0.4 \
  --max_gradient_norm=3.0 \
  --attention=scaled_luong \
  --num_units=512 \
  --num_gpus=1 \
  --batch_size=512 \
  --src=miz --tgt=ferp \
  --vocab_prefix=$TRDIR/vocab \
  --train_prefix=$TRDIR/train1 \
  --dev_prefix=$TRDIR/dev \
  --test_prefix=$TRDIR/test \
  --out_dir=$MODDIR/model \
  --num_train_steps=20000 \
  --steps_per_stats=20000 \
  --steps_per_external_eval=20000 \
  --tgt_max_len=180 \
  --tgt_max_len_infer=280 \
  --src_max_len=100 \
  --num_layers=2 \
  --dropout=0.2 \
  --metrics=bleu \
  --beam_width=240 \
  --num_translations_per_input=240 > $TRDIR/train.log-tr-1 
```

## Generate Programs (Inference)

```bash
time python -m nmt.nmt --beam_width=240 --num_translations_per_input=240 --out_dir=$MODDIR/model --inference_input_file=$inp --inference_output_file=$out >> $TRDIR/train.log-inf;
```

`$inp` is a file with the OEIS sequences for which we want program predictions encoded like this:

```
_ # _ # _ # _ # _ # _ # 1 2 6 0 # 5 #
~ 1 2 8 # ~ 1 2 8 # 1 9 2 # ~ 1 2 8 # 3 2 # 3 2 # ~ 4 8 # 3 2 # ~ 8 # ~ 8 # 1 2 # ~ 8 # 2 # 2 # ~ 3 # 2 #
1 # 1 # 1 # 1 # 1 # 1 # 1 # 0 # 1 # 1 # 1 # 0 # 0 # 1 # 0 # 0 #
7 5 # 6 6 # 5 7 # 4 8 # 4 1 # 3 4 # 2 7 # 2 2 # 1 7 # 1 2 # 9 # 6 # 3 # 2 # 1 # 0 #
```

We only use 16 input values (this is arbitrary) and turn any values with more than 6 digits into "_" . "-" (minus) becomes "~" .

We also reverse the sequence (probably not needed with our current NMT params).

## Validating and Submitting Programs

Add a generator with `"version": 8` to the LODA `miners.json` config. Set the `batchFile` to the file that contains your generated programs. Add a new miner config (`nmt` below) and reference the new generator by its name.

```json
{
  "miners": [
    {
      "name": "nmt",
      "overwrite": "none",
      "enabled": true,
      "backoff": true,
      "generators": [
        "v8"
      ],
      "matchers": [
        "direct",
        "linear1",
        "linear2",
        "binary",
        "decimal"
      ]
    },
  ],
  "generators": [
    {
      "name": "v8",
      "version": 8,
      "batchFile": "/path/to/oeis-loda-nmt-predictall"
    }
  ]
}
```

Run the validation (if you run in `client` mode, found programs will be automatically submitted):

```bash
loda mine -i nmt
```

To parallelize it, you need to edit the `miners.json` file:

* create multiple "generators" where each point to a different chunk
* create multiple "miners" (with different names) each pointing to another generator

To limit the evaluation time per program, use `-z` option. To allow updating existing programs (slower), set `overwrite` to `auto` or `all`.
