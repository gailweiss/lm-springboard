import lightning as pl
import torch
import wandb
from time import process_time
from misc.util import printer_print as print


class Trainer(pl.LightningModule):
    def __init__(self, model, train_params, start_time=None,
                 samples_at_validation=True,
                 train_dataloader_nbatches=None):
        super().__init__()
        self.model = model
        self.train_params = train_params
        self.curr_train_stats_by_type = {"loss": {}, "acc": {}}
        self.curr_val_stats_by_type = {"loss": {}, "acc": {}}
        self.start_time = start_time  # should be obtained with process_time()
        self.n_train_samples = 0
        self.n_train_batches = 0
        self.n_opt_steps = 0
        self.stat_counter = -1
        self.logged_stats_dict = {}
        self.samples_at_validation = samples_at_validation
        # allows turning sampling off in specific cases,
        # in particular when making trainers for quick validation checks
        # through model_explorer.py
        self.last_val_loss = None
        # for reading the val loss after initiating a validation check with
        # lightning, used by model_explorer.py
        self.automatic_optimization = False
        # gain more control of optimization - lightning automatic optimization
        # quite constrained in options
        self.last_checkpoint_i = -1
        self.last_checkpoint_nsamples = -1
        self.stat_syncer = 0
        self.curr_epoch = -1
        self.val_count_in_epoch = -1
        self.train_dataloader_nbatches = train_dataloader_nbatches
        self.log_stat("n_train_samples", self.n_train_samples)
        self.log_stat("n_train_batches", self.n_train_batches)
        self.log_stat("n_opt_steps", self.n_opt_steps)

    def prepare_saver(self, dp, saving_folder, saving_function):
        self.dp = dp
        self.saving_folder = saving_folder
        self.saving_function = saving_function

    def save_checkpoint(self):
        if self.n_train_samples == self.last_checkpoint_nsamples:
            return  # already saved this one, e.g. coming here from epoch end
            # after just having saved by other means
        fn = f"{self.saving_folder}/{self.n_train_samples}"
        self.saving_function(fn, self.trainer, self, self.model.model_params,
                             self.dp, self.train_params)
        self.last_checkpoint_nsamples = self.n_train_samples

    def maybe_save_checkpoint(self, after_val=False, after_train_epoch=False):
        if self.train_params.checkpoint_every == 0 or\
           self.train_params.checkpoint_every < -1:
            return  # never checkpoints

        if self.train_params.checkpoint_every > 0:
            checkpoint_i = (self.n_train_samples //
                            self.train_params.checkpoint_every)
            if checkpoint_i > self.last_checkpoint_i:
                self.save_checkpoint()
                self.last_checkpoint_i = checkpoint_i
                return

        if after_val and self.train_params.checkpoint_every == -1:
            self.save_checkpoint()
            return
        if after_train_epoch:
            self.save_checkpoint()

    def log_stat(self, name, val):
        if not self.train_params.no_wandb:
            wandb.log({name: val})
        if name not in self.logged_stats_dict:
            self.logged_stats_dict[name] = []
        self.stat_counter += 1  # increases with every single stat,
        # unlike stat syncer, which increases exactly once at the beginning
        # of every training step
        self.logged_stats_dict[name].append((self.stat_syncer,
                                             self.n_train_samples,
                                             self.stat_counter,
                                             val))

    def log_time(self):
        if None is not self.start_time:
            self.log_stat("process_time", process_time() - self.start_time)

    def on_train_epoch_start(self):
        self.curr_epoch += 1
        self.val_count_in_epoch = -1
        print("starting epoch:", self.curr_epoch)
        self.logged_epoch_count_yet = False

    def on_train_epoch_end(self):
        # note that averaging accs will give "average batch accuracy" but not
        # actual full dataset accuracy (as batches may have different numbers
        # of tokens)
        self.log_time()
        for sn in self.curr_train_stats_by_type:
            d = self.curr_train_stats_by_type[sn]
            for t, stats in d.items():
                self.log_stat(f"stat/train_{sn}:{t}", wary_mean(stats))
            self.curr_train_stats_by_type[sn] = {}
        self.maybe_save_checkpoint(after_train_epoch=True)
        self.log_stat("n_epochs", self.curr_epoch)

    def on_validation_epoch_end(self):
        # note that averaging accs will give "average batch accuracy" but not
        # actual full dataset accuracy (as batches may have different numbers
        # of tokens)
        self.val_count_in_epoch += 1
        self.log_time()
        main = self.curr_val_stats_by_type["loss"]["main"]
        for sn in self.curr_val_stats_by_type:
            d = self.curr_val_stats_by_type[sn]
            for t, stats in d.items():
                self.log_stat(f"stat/val_{sn}:{t}", wary_mean(stats))
            self.curr_val_stats_by_type[sn] = {}

        val_loss = wary_mean(main)

        print(f"\n {'='*20} \n epoch [{self.curr_epoch}] val cycle",
              f"[{self.val_count_in_epoch}] val loss: [{val_loss}]",
              f"\n {'='*20} \n ")
        self.last_val_loss = val_loss  # might want this e.g. in model_explorer
        if self.samples_at_validation:
            print("=== sampling: ===")
            sample = self.model.sample(
                        max_seq_len=self.train_params.max_sample_tokens,
                        temperature=self.train_params.sample_temperature)
            # linux doesnt always like printing the samples if they have funny
            # characters
            try:
                print(sample)
            except Exception as e:
                print("couldn't print the sample here :(")
                print(e)
            print("\n")
        self.maybe_save_checkpoint(after_val=True)

    def record_type_stats(self, stats, recording_dict, from_train=False,
                          stat_name="loss"):
        # dont want to record losses with their graphs
        def item(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            else:
                return v

        for t in stats:
            if t not in recording_dict:
                recording_dict[t] = [item(stats[t])]
            else:
                recording_dict[t].append(item(stats[t]))
        if from_train:
            for t in stats:
                self.log_stat(f"stat/train_batch_{stat_name}:{t}",
                              item(stats[t]))

    def curr_avg_lr(self):
        lrs = [pg['lr'] for pg in self.lr_schedulers().optimizer.param_groups]
        return -1 if not lrs else wary_mean(lrs)

    def log_hyperparams_and_time(self):
        n_active_params = sum(p.numel() for p in
                              self.model.parameters() if p.requires_grad)
        self.log_stat("n_active_params", n_active_params)
        self.log_time()

    def maybe_log_hyperparams_and_time(self):
        freq = self.train_params.hyperparams_log_freq
        if self.n_opt_steps % freq == 0:
            self.log_hyperparams_and_time()

    def on_train_batch_start(self, batch, batch_idx):
        if self.train_params.early_stop_nsamples > 0:
            if self.n_train_samples >= self.train_params.early_stop_nsamples:
                return -1
                # will stop the training, according to pytorch lightning

    def training_step(self, batch, batch_idx):
        self.stat_syncer += 1
        self.log_stat("stat_syncer", self.stat_syncer)
        self.maybe_log_hyperparams_and_time()
        self.maybe_save_checkpoint()
        clear_gpu_caches()

        a = self.model.get_losses(batch, accs_too=True)
        losses, accs, n_samples = a["loss"], a["acc"], a["n_samples"]
        

        for sn in ["loss", "acc"]:
            self.record_type_stats(a[sn], self.curr_train_stats_by_type[sn],
                                   from_train=True, stat_name=sn)

        self.log("train_batch_loss", losses["main"].item())
        # for the lr scheduler

        self.log_stat("avg_lr", self.curr_avg_lr())
        self.log_stat("n_train_samples", self.n_train_samples)
        self.log_stat("n_train_batches", self.n_train_batches)
        if not self.logged_epoch_count_yet:
            self.log_stat("n_epochs", self.curr_epoch)
            self.logged_epoch_count_yet = True
        self.log_stat("n_opt_steps", self.n_opt_steps)
        self.log_stat("weight_norms", self.get_weight_norms())

        self.manual_backward(losses["main"])
        self.maybe_step_opt_and_lr(batch_idx)
        # update counters *after* logs, for more logical record:
        # first loss is after 0 batches, not 1
        self.n_train_batches += 1
        self.n_train_samples += n_samples

    def maybe_step_opt_and_lr(self, batch_idx):
        if (batch_idx + 1) % self.train_params.accumulate_grad_batches == 0:
            opt = self.optimizers()
            self.clip_gradients(
                opt, gradient_clip_val=self.train_params.gradient_clip_val,
                gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            sched = self.lr_schedulers()
            sched.step(self.trainer.callback_metrics["train_batch_loss"])
            self.n_opt_steps += 1

    def validation_step(self, batch, batch_idx):
        a = self.model.get_losses(batch, accs_too=True)
        n_samples = a["n_samples"]
        for sn in ["loss", "acc"]:
            self.record_type_stats(a[sn], self.curr_val_stats_by_type[sn],
                                   stat_name=sn)
        return a["loss"]["main"].item()

    def make_main_scheduler(self, optimizer):
        if self.train_params.lr_scheduler_type == 'Plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau
            return sched(optimizer, mode='min', verbose=False,
                         min_lr=self.train_params.min_lr,
                         factor=self.train_params.scheduler_factor,
                         patience=self.train_params.patience)
        elif self.train_params.lr_scheduler_type == 'Cyclic':
            sched = torch.optim.lr_scheduler.CyclicLR
            return sched(optimizer, base_lr=self.train_params.min_lr,
                         max_lr=self.train_params.lr,
                         step_size_up=self.train_params.lr_cycle_steps // 2,
                         mode='triangular',
                         gamma=self.train_params.scheduler_factor,
                         cycle_momentum=False)
        elif self.train_params.lr_scheduler_type == 'Cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR
            return sched(optimizer, self.train_params.lr_cycle_steps,
                         eta_min=self.train_params.min_lr)
        elif self.train_params.lr_scheduler_type == 'Linear':
            sched = torch.optim.lr_scheduler.LinearLR
            expected_main_scheduler_steps = (
                (self.train_params.epochs * self.train_dataloader_nbatches) //
                self.train_params.accumulate_grad_batches
            ) - self.train_params.lr_warm_steps
            end_factor = self.train_params.min_lr / self.train_params.lr
            return sched(optimizer, start_factor=1.0,
                         end_factor=end_factor,
                         total_iters=expected_main_scheduler_steps)
        else:
            raise Exception("unknown scheduler type:",
                            self.train_params.lr_scheduler_type)

    def reconfigure_optimizers(self):
        optimizers, _ = self.configure_optimizers(
            existing_scheduler=self.lr_schedulers())
        self.optimizers()._optimizer = optimizers[0]

    def get_optimizer_params(self, weight_decay):
        # note: if ever add freezing into this code, will have to update
        # code here to avoid sending frozen parameters into the optimizer:
        # e.g. i think optimizers with weight decay will apply it even to
        # frozen parameters
        if weight_decay <= 0:
            return self.parameters()
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if self.model.in_main_part(name) and \
               self.model.not_layernorm(name) and \
               not name.endswith("bias"):
                # each class uses different parameter names,
                # best to have them self report what should be weight decayed
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    def configure_optimizers(self, existing_scheduler=None):
        weight_decay = self.train_params.weight_decay

        optimizer_params = self.get_optimizer_params(weight_decay)
        optimizerClass = torch.optim.AdamW if weight_decay > 0 else \
            torch.optim.Adam
        optimizer = optimizerClass(optimizer_params, lr=self.train_params.lr)

        def f_warmup(n):
            assert n <= self.train_params.lr_warm_steps
            return n / self.train_params.lr_warm_steps

        s_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, f_warmup)
        s_main = self.make_main_scheduler(optimizer)
        s_full = MyChainedScheduler() if existing_scheduler is None else \
            existing_scheduler
        s_full.setup(optimizer, [s_warmup, s_main],
                     milestones=[self.train_params.lr_warm_steps])
        # get scheduler started, else first batch has max value apparently
        s_full.step(None)
        s_main = {"scheduler": s_full}

        return [optimizer], [s_main]

    def get_weight_norms(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


class MyChainedScheduler:
    def __init__(self):
        pass

    def setup(self, optimizer, schedulers, milestones):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones  # the # steps after which to switch

        for i in range(len(milestones) - 1):
            assert milestones[i] < milestones[i + 1]

        self.num_steps = 0

    def get_curr_scheduler(self):
        curr_scheduler_i = next((i for i, e in enumerate(self.milestones)
                                if self.num_steps < e), len(self.milestones))
        return self.schedulers[curr_scheduler_i]

    def step(self, metric):
        curr_scheduler = self.get_curr_scheduler()
        self.num_steps += 1
        if isinstance(curr_scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            curr_scheduler.step(metric)
        else:
            curr_scheduler.step()

    def state_dict(self):
        return self.get_curr_scheduler().state_dict()

    def load_state_dict(self):
        return self.get_curr_scheduler().load_state_dict()


def clear_gpu_caches():
    if torch.cuda.is_available:
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def wary_mean(vals):
    vals = [v for v in vals if None is not v]
    if not vals:
        return None
    return sum(vals) / len(vals)
