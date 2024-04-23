# Hello

This repository has all the boilerplate for making changes to the transformer architecture and training the modified model, so that you can dive directly into the logic of your idea. It allows evaluating the architectures on some natural and some synthetic tasks, as well as implementing or providing new tasks.

At its base, this repository trains a transformer on a task of your choice, with the help of pytorch lightning, and tracks metrics with wandb. It surfaces all the relevant code so that you can go directly to defining or adding additional tasks, editing the transformer architecture or training behaviour, and comparing your modified methods to the originals. There's nothing special to any of the code here, but if you haven't already stuck it all together, this repository is all about skipping that step and going straight to your idea. (For me: it's the base repository from which I start every time I have a "what if the transformer architecture was actually..." idea).

### Notes

This repository also lets you poke around GPT2-small as taken from huggingface, albeit only in a limited interface that is adapted to match the one for the models you will train here. If your main goal is to poke around open source pretrained models - and with a much richer interface than here - then you probably want Neel Nanda's TransformerLens instead: [https://github.com/neelnanda-io/TransformerLens].

For now, this repository is only concerned with the autoregressive language modeling objective - i.e., decoders, as opposed to encoders or encoder-decoders (think GPT models as opposed to BERT or T5). There is one model wrapper: the LM (language model), and one trainer: the LMTrainer. 

This repository is missing many things: base implementations of more architectures (e.g. RNNs); training for objectives other than autoregressive language modeling; using beam search to sample from the models; getting more pretrained models; getting attention from these pretrained models; etc. Expansions are welcomed :)

# Contents

In the Basics sections, you can follow along a minimal example with things already in this repository: in `Plain Run`, you can train a transformer, track its loss, and save it. In `Basics: Inspect Model`, load it (or GPT2-small if you prefer), and inspect your loaded model. In `Basics: Configs`, gain some control: change hyperparameters, give extra command line args, and run over multiple configurations.

Once that's done, get into your logic! In `Custom Datasets` see how to define or add your own tasks, and in `Customising` see some examples of how to inject your logic into this code. There's nothing too exciting in these examples, but they should give you a clear idea of where to go to insert your own more interesting modifications :)

# Basics: Setup, and a Plain Train Run

## Requirements

The full requirements list is in `requirements.txt`, which lists also the versions of the relevant packages that I happen to be using - though you will probably have success with your versions too.

For a small rundown of the parts being used: this repository builds/uses neural networks in pytorch (`torch`); trains them with pytorch lightning (`lightning`); logs their losses and other statistics with wandb (`wandb`); and displays their attentions or logged training stats (when requested) with matplotlib (`matplotlib`). It allows coupling the neural networks with either custom tokenizers, or pretrained ones, the latter taken from huggingface (`transformers` and `tokenizers`). In addition to using local files, it allows getting some small language modeling tasks from huggingface (`datasets`). It also allows loading gpt2-small from huggingface (`transformers`). 

## Plain Run

For your first call, open the terminal and try: 

```
python3 main.py --config=test-nlp --task=ptb --no-wandb
```

The code will very quickly train a tiny transformer - whose configuration is described in `configs/test-nlp.txt` - on the Penn Treebank. It will print samples at the end of every epoch (albeit bad ones, as this is just a short dummy train).

## Setup and Toggle WandB

If you want to track your models' training in a nice interface, you may want to set up wandb, which this repository has integrated. Follow the instructions at `wandb.ai` to set up an account and connect it your computer. Then, open `main.py`, find the `wandb_username` variable (line 40), and put in your username. 

You can now run the command above again, this time without `--no-wandb`, to train a model and track its progress in wandb. 

```
python3 main.py --config=test-nlp --task=ptb
```

The run will be named randomly by wandb and directed to a project `base-test-nlp-ptb` in your account on `wandb.ai`.

## Save Models

If you want a model to save after training, add the argument `--save` to your main call: 

```
python3 main.py --config=test-nlp --task=ptb --save
```

The model will be saved in a subfolder of `../saved-models` relative to the location of this code, in this case specifically: `../saved-models/test-nlp/ptb/{random wandb name}/{timestamp}`, where `{random wandb name}` is the name assigned by wandb to the run, and `{timestamp}` describes its starting time. The model will be saved alongside its tokenizer, the full configuration used to create it, and various statistics tracked during its training, such as the loss of every training batch.

# Basics: Inspect Model
## Load Saved Model

To load a model you have saved, open your terminal or a jupyter notebook, and run
```
import model_explorer
timestamp = {timestamp}  # your timestamp here
lm, dataset, train_stats, params = model_explorer.get_model_by_timestamp(timestamp)
```
where `{timestamp}` is the timestamp of your saved model as it appears in its containing folder's path (see above), e.g. `timestamp = "2024-04-22--16-17-00"`. 

This function returns 4 values, `lm`, `dataset`, `train_stats`, and `params`. 

- `lm` is an `LM` as defined in `models/lm.py`, you can see examples on how to use it below. 
- `dataset` is an `LMDataModule` as defined in `data/dataloader.py`, it can return torch dataloaders with the functions `train_dataloader`, `test_dataloader`, and `val_dataloader`, each of which receive a requested batch size. You can also see a nice print of each of its samples with the function `show_sample`, which expects a sample index - e.g. `dataset.show_sample(1)`. 
- `train_stats` is a dictionary of stats logged in training, it is described below (Show Training Metrics). 
- `params` is a dictionary of the different parameters used for the model architecture, training method, and training data. 

### Verify Loaded Well (or at least sanity check)

The last thing that `main.py` does before saving a model is check and store its validation loss. To rule out any obvious mistakes, we can make sure that the model we have loaded still obtains the same validation loss on its data as it claimed to when saved, behaves deterministically, and so on. Run 

```
model_explorer.verify_stable_load(timestamp)
``` 

to do some basic checks, and see that it doesn't complain. 

### Show Training Metrics

Independently of wandb logging, saved models also save their recorded losses, and other metrics. 

Run 

```
model_explorer.plot_metric(train_stats,"validation_loss")
```

to see the validation losses over time (measured in number of trained samples) of the model you have loaded. You can also plot any other metric in `train_stats`, you can list these by running `list(train_stats.keys())`

You can also save this plot by passing the argument `folder_name` to `plot_metric`. It will save the plot in `../metrics/{folder_name}/{metric_name}.png` relative to `main.py`.

In case you wish to plot something more complicated, here is a description of `train_stats` to help you: 

`train_stats` is a dictionary of all the metrics tracked in `lmtrainer.py` with the function `log_stat` during training training. Its keys are the names of all these metrics. Each individual metric is stored as a list of tuples `(n,v,c)` as follows: `v` is a value of the metric, `n` is the number of samples that had been trained on up until recording this value, and `c` is the stats counter - the number of values stored in `train_stats` (across all metrics) when this tuple was added. These lists of tuples are each sorted in increasing order of `c`. 

## Load GPT2-small (Rudimentary for now)

You can load gpt2-small in here, though the interface is very limited for now.

```
import gpt2
lm_gpt2 = gpt2.get_gpt2()
```

Cons: 
- The code here doesn't take advantage of all the nice optimisations and options huggingface has provided. If you just use huggingface to load it (which is also very straightforward) you will get a model on which you can sample sequences efficiently and with various algorithms, whereas here you just have the one greedy sampler.
- I haven't implemented getting GPT2's attention. I will really appreciate someone expanding this so I can get attention from GPT2-small as easily as I can get it from my own models. See the `forward` function in `model/lm.py` to see where to insert such code.

Pros: 
- Except for inspecting attention, which is not yet implemented, you can use this to interact with gpt2 through the same interface as provided for the models trained here.

Overall, if you want to poke around open source pretrained models, you may prefer existing libraries for mechanistic interpretability - e.g. TransformerLens: [https://github.com/neelnanda-io/TransformerLens/]

## Inspect Model
### Generate

Sample from your loaded model `lm` (or `lm_gpt2`) with the `sample` function, which allows receiving an initial prefix `pref` (string or list of token ids), a maximum sample length `max_seq_len` (in tokens, default 100), a sampling temperature `temperature` (default 1), and returning the sample either as a string or list of indices with the parameter `as_str` (default `True`). 
```
lm_gpt2.sample(pref="Well the weather for the whole area will",max_seq_len=50,temperature=0.5,as_str=True)
lm_gpt2.sample(pref=[50256, 4053, 262, 6193])
lm.sample(pref="")
```
### Get Outputs

Get the output embeddings of your language model on a batch of inputs (presented as a tensor with shape batch size x seq len) or single input (presented as a string, list of ids, 1-D tensor, or batch of size 1 as above) by calling it directly:

```
import torch 
a = lm_gpt2("hello") # string
b = lm_gpt2(lm_gpt2.tokenizer("hello")) # list of indices
assert torch.equal(a,b)

c = lm_gpt2(torch.Tensor(lm_gpt2.tokenizer("hello"))) # 1-D tensor
assert torch.equal(a,c)
d = lm_gpt2(torch.Tensor(lm_gpt2.tokenizer(["hello","hi"]))) # batch
assert False not in torch.isclose(a,d[0:1])
``` 

### Compute Perplexity (or Cross Entropy Loss)

Compute the perplexity (or cross entropy loss) of your language model on a set of sequences using its `perplexities` function, which computes the mean, maximum, and minumum per-token perplexity (or CE loss), and notes also how many tokens were considered for the computation. If setting `per_token=True` it also returns the perplexity (or CE loss) at every position, with shape (batch size X seq len -1), otherwise (if setting `per_token=False`) this value is an empty list. Set `before_exp=True` to get CE losses instead of perplexity.

The input sequences can be presented either as a list of sample strings, or a torch DataLoader.

For examples:
```
mean_p, max_p, min_p, total_tokens, per_token_res = lm_gpt2.perplexities(["hello","hi"],per_token=True)
```
or
```
mean_l, max_l, min_l, total_tokens, per_token_res = lm.perplexities(dataset.test_dataloader(16),per_token=True,before_exp=True)
```

When applied to several sequences of different lengths, `per_token_res` is shaped according to the maximum sequence length (minus 1), and holds `dummy_res` (default -1) wherever no prediction was made.

### Inspect Attention

Show the attention patterns of a transformer lm using the `model_explorer` function `show_lm_attns`. This function will work with the trainable transformers in this repository, if you add non-transformer layers however, you will have to edit the relevant logic in `model/transformer/transformer.py`. This function will not work on GPT2-small, but I will appreciate someone adding that in. 

```
out_embeds, attns = model_explorer.show_lm_attns(lm,"as business managers know")
```

`show_lm_attns` expects as input an LM object and a single sequence, in the same possible formats that the LM accepts for a single sequence (i.e. string, list of indices, or 1D tensor). It will show the attention patterns of each head in each layer on that sequence, or subsets of the heads and layers as specified by the arguments `layers` and `heads`. It can save these as images in a folder `../attentions/{folder_name}` if the argument `folder_name` is specified:

```
out_embeds, attns = model_explorer.show_lm_attns(lm,lm.tokenizer("as business managers know"),layers=[0],heads=[0],folder_name="demo")
```

This function also returns `out_embeds` and `attns`, which give the model's output logits and attention patterns on this input sequence as their names imply. Their shapes are `batch size (i.e. 1) x seq len x vocab size` and `batch size (i.e. 1) x n layers x n heads x seq len (in) x seq len (out)`, respectively.

# Basics: Configs
## Command Line Args

The shortest command to train a model is `python3 main.py --config={config_name}`, where `{config_name}.txt` is a config file placed in the `configs/` folder; the contents of this file are described in the next section. The full set of args to the main call are:

`--config`: the name of the config file to use, e.g. `--project=test-nlp`, excluding the extension `.txt`. If there exist also config files named `{config}-{*}.txt`, these will also be run. For example running `python3 main.py --config=test` will run all experiments outlined in `configs/test-nlp.txt` and in `configs/test-synth.txt`, but `python3 main.py --config=test-nlp` will run only those in `test-nlp.txt`.

`--task`: Overrides the task specified in the config file.

`--wandb-proj-name`: informs the project name under which wandb will store all the runs created by this call. Relevant if wandb is being used (default: yes). The full project name wandb will use is specified by the `wandb_proj_name` function of the `Namer` class in `main.py`. Defaults to the value set in `--config`.

`--save`: if set, store the models trained by this call. They will be kept in a subfolder of `../saved_models`. The full (sub)path of each such subfolder is specified by the `save_folder_name` function of the `Namer` class in `main.py`, and includes a timestamp of when the model started running.

`--ablate`: if set, will run also ablations on the configs described in the config files. By default, no such ablations are defined, but they can be added to the `run_all` function of `main.py`.

`--no-wandb`: turn off wandb tracking for these runs. The `train_stats` dictionary will still be created and stored if saving the models.

`--gpu-id`: set this all to run on a specific gpu of your computer, provided these are available. e.g. `--gpu-id=1`


## Config Files

### Basic

Config files are stored in the `configs/` folder, it contains a few initial examples, and you can add your own. They specify the values for the different parameter (group)s related to a training run: `DataParams`, `TrainParams`, and `ModelParams`. The full set of possible arguments for these parameters are defined in the files `data/dataloader.py`, `model/train_params.py`, and `model/model_params.py` respectively, along with their default values and a short description of their meanings. 

The values in the config files are grouped under headers marking which set of parameters they belong to - DataParams, TrainParams, or ModelParams - see the provided config files for an example.

### Running Multiple Configs

You can define multiple configurations in a single config file by giving a list of values for some parameters, e.g. by setting dim=[256,512] in your config file. If multiple parameters are given multiple values, main.py will loop over the cartesian product of all the given options in the file - e.g., setting dim=[256,512] and lr=[1e-3,3e-4] will yield 4 training runs. 

Additionally, you may set main.py to run through multiple config files, by naming them all with in the format `{shared-prefix}-{specifics}.txt` (e.g. `small-1.txt`, `small-2.txt`) and setting `--config` to that shared prefix (e.g. `--config=small`).

# Custom Datasets
## From File

You can add a custom dataset from a file by placing it in a local data folder and directing this code to that folder. Place a file `data-path.txt` in `../../data-path.txt` relative to `main.py`, containing the path to your local data folder. Place your custom dataset in a text file `data.txt`, one line per sample, in a subfolder `{task_name}` of your local data folder. You can then set `dataset_name={task_name}` to load that file as your data. Alternately, you can implement custom logic to load your dataset, and add a call to it from the `get_data` function of `data/dataloader.py`.

## Define Synthetic Dataset

You can define a synthetic task by implementing and "registering" a function generating random samples for that task in `data/syntheticdata.py`. Decorate the function with `@registered` to register it, you will then be able to call that task by the name of this function. See e.g. `copy` and `histogram` in `data/syntheticdata.py` for examples. Note that, as implemented here, synthetic tasks only work with char-level tokenizers. You will have to explicitly change this behaviour if you want something else.

# Customising

What we're here for! Here we will show how to use this code to define and evaluate various modifications to the transformer architecture. This section is done by examples.

## Note: Naming: Project; WandB runs; Save Folder

The wandb project and runs your experiment logs will be saved in, and the folders any saved models will be stored in, are named by the relevant functions of the `Namer` class in `main.py`. Each run has a Namer holding all of its parameters (data, model, and train), the command line arguments with which `main.py` was called, and a timestamp from when it started. You can edit the Namer's functions to make informative run and save folder names for your experiments. As a default, the wandb project and save folders these runs will be stored in include the `MAIN_PROJ`value set in `main.py` (default: `"base"`).

Let's begin by setting `MAIN_PROJ="edit-examples"` in `main.py` to mark this set of edits for all of our saved wandb logs and folders.

## Examples: Model Architecture
### Example: Inserting a ReLU layer after the first encoder layer

Here we will add an extra ReLU layer after the first TransformerEncoderLayer in our transformer. We will add an argument to the model to toggle this change on or off.
1. Open `model/model_params.py` and add an argument `inject_relu: bool = False` to the ModelParams dataclass.
2. Open `model/transformer/transformer.py` and change the assignment of `self.layers` in the `__init__` function of the `Transformer` class. Specifically, replace the line 

```
self.layers = nn.ModuleList([make_layer() for _ in range(self.model_params.n_layers)])
```

with the lines

```
tlayers = [make_layer() for _ in range(self.model_params.n_layers)]
if self.model_params.inject_relu:
	tlayers = tlayers[0] + [nn.ReLU()] + tlayers[1:]
self.layers = nn.ModuleList(tlayers)
```

3. You can now create a config file with `inject_relu=[True,False]` under the `ModelParams` section to train models with identical configurations, once with and once without this extra layer. 
4. Naming: To make it easier to tell the difference between these runs, update the `run_name` function in `Namer` in `main.py`. For example, replace `return None` with `return f"with-relu-layer:[{mp.inject_relu}]--{model_str}"`. `save_folder_name` will default to including the output of `run_name`, so it is not necessary to update.

### Example: Squaring the Feed-Forward SubLayer's output

We're going to go deeper and modify the actual TransformerEncoderLayer.
1. Open `model/model_params.py` and add an argument `square_ff: bool = True` to the `ModelParams` dataclass. At this point, you may also want to set the default value of `layer_architecture` to `"custom-transformer"` - your edits will only be accessed when `layer_architecture` is set to this value.
2. Open `model/transformer/transformerencoderlayer.py` and go to the function `_ff_block`. Replace the line 

```
return self.dropout_ff(x)
```

with the lines

```
res = self.dropout_ff(x)
return torch.pow(res,2) if self.model_params.square_ff else res
```

3. As in 3 and 4 in the example above, you can now run experiments on configurations that are identical up to the inclusion of this feed forward layer squaring, and give them informative names. Make sure to set `layer_architecture="custom-transformer"` through these configurations (or to update its default value in the `ModelParams` class and leave it unspecified in the configurations), or the code will use `nn.TransformerEncoderLayer` from the torch library instead of your customised implementation.

### Example: Multiply the Attention Scores by a Given Scalar

Now we're going to modify the attention computation itself. Specifically, we will multiply the attention scores - before the softmax - by a given scalar.

1. Open `model/model_params.py` and add an argument `attn_scalar: int = 1` to the ModelParams dataclass. Again, you will want either to set the default of `layer_architecture` to `"custom-transformer"` here, or explicitly note this in your configuration files, for this change to go through.
2. Open `model/transformer/torch_f_multi_head_attention_forward.py` and go to the line where the attention scores are about to be softmaxed:

```
attn_output_weights = softmax(attn_output_weights, dim=-1)
```

immediately before that line, insert:

```
attn_output_weights = attn_output_weights * model_params.attn_scalar
```

3. As in 3 in the example above: remember to set `layer_architecture="custom-transformer"` to use this change, and then try some different values of attention score scaling by creating a config file with (for example) `attn_scalar=[1,2,3]`.

### Example: Multiply the Attention Scores by a Given Scalar, but only when the batch contains token id 10

Maybe you want to edit the attention computation differently in different cases, i.e., the static values of `model_params` are not enough to describe everything that you want to change in the function. For this, the custom attention function also receives the argument `attn_requests`. We'll repeat the example above, but this time condition the scaling on the presence of token id 10 in the batch.

In this example, we will treat the `attn_requests` argument as a simple boolean marking whether to apply the scaling or not.

1. Implement the attention scaling option as in the example above.
2. Open `model/lm.py` and go to the `forward` function. Just after the line 

```
assert cond, msg
```
, insert the line:

```
attn_requests = 10 in x
``` 

3. Open `model/transformer/torch_f_multi_head_attention_forward.py` again, and condition the scaling on the attn_requests: replace the line

```
attn_output_weights = attn_output_weights * model_params.attn_scalar
```

with the lines

```
if attn_requests:
	attn_output_weights = attn_output_weights * model_params.attn_scalar
```

4. As with (3) in the example above - try out the configurations, and make sure to set `layer_archicture="custom-transformer"`.


## Examples: Train Loop

Maybe you also have ideas for changes to the train loop, or additional metrics you would like to track. Let's see some examples.

### Example: Freezing Random Layers

Let's freeze random layers at every train batch. Note: I am not sure how this will interact if stepping on multiple batches at a time, so set `accumulate_grad_batches=1` in the TrainParams for this one (this parameter is described in `model/train_params.py`).

1. Open `model/train_params.py` and add an argument `freeze_random_layers: bool = False`.
2. Open `model/lmtrainer.py` and go to the function `get_loss`. Just before the line

```
z = self.lm(x)
```

insert the lines:

```
if self.train_params.freeze_random_layers:
	for layer in self.lm.decoder.layers:
		layer.requires_grad_((torch.randint(2,(1,)).item() == 1))
```
This will freeze a random subset of the layers.

3. Try it out as with the example above, setting informative run names. This modification does not affect the architecture and so can run with `layer_architecture="torch-transformer"` and enjoy the optimised torch implementations.

### Example: Corrupting Input (but not Target) During Training

1. Open `model/train_params.py` and add an argument `corruption_frac: float = 0.0`.
2. Open `model/lmtrainer.py` and again go to the function `get_loss`. Just before the line 
```
z = self.lm(x)
```

insert the lines
```
if from_train and self.train_params.corruption_frac > 0:
	x = self.apply_corruption(x)
```

Then add a function `apply_corruption` to the `LMTrainer` class as follows:
```
	def apply_corruption(self,x):
		bsz,seq_len = x.shape
		dropout = torch.nn.Dropout(self.train_params.corruption_frac)
		rand_replacement = torch.randint(0,self.lm.n_tokens,x.shape).to(device=x.device)
		corr_locs = (dropout(torch.ones(x.shape))==0).to(device=x.device)
		return torch.where(corr_locs,rand_replacement,x)
```

3. Try it out as in the example above

### Example: Tracking Maximum Parameter Value, but only every now and then.

1. In case you will want to turn this off, optionally add an argument `track_max: bool = True` in `model/train_params.py`
2. In `model/train_params.py`, set `hyperparams_log_freq` to the frequency you would like to log this and other hyperparameters at - the number of batches to train between each logging of this metric.

3. Open `model/lmtrainer.py` and navigate to the function `log_hyperparams_and_time`. Assuming you have added the argument `track_max` into `TrainParams`, add the lines:

```
if self.train_params.track_max:
	max_param_val = max(p.max().item() for p in lm.parameters())
	self.log_stat("max_param_val",max_param_val)
```

into this function.

4. This parameter will now (when requested through `track_max`) be tracked, and will show up in any wandb runs and appear in the `train_stats` dictionary of saved models.

5. If you want to track this metric at *every* training batch, navigate instead to the function `get_loss` in `model/lmtrainer.py` and add the lines there instead:

```
if self.train_params.track_max and from_train:
	max_param_val = max(p.max().item() for p in lm.parameters())
	self.log_stat("max_param_val",max_param_val)
```
