import lightning as pl
import torch
import wandb
from time import process_time


class LMTrainer(pl.LightningModule):
    def __init__(self, lm, train_params, start_time=None,
                 samples_at_validation=True):
        super().__init__()
        self.lm = lm
        self.train_params = train_params
        self.ignore_index = self.lm.tokenizer.pad_token_id
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.curr_val_losses = []
        self.curr_train_losses = []
        self.curr_steps_count = 0
        self.this_lm_total_batches = 0
        self.start_time = start_time  # should be obtained with process_time()
        self.n_train_samples = 0
        self.stat_counter = -1
        self.logged_stats_dict = {}
        self.samples_at_validation = samples_at_validation
        # allows turning sampling off in specific cases,
        # in particular when making lmtrainers for quick validation checks
        # through model_explorer.py
        self.last_val_loss = None
        # for reading the val loss after initiating a validation check with
        # lightning, used by model_explorer.py

    def log_stat(self, name, val):
        if not self.train_params.no_wandb:
            wandb.log({name: val})
        if name not in self.logged_stats_dict:
            self.logged_stats_dict[name] = []
        self.stat_counter += 1
        self.logged_stats_dict[name].append((self.n_train_samples,
                                             val,
                                             self.stat_counter))

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.maybe_log_hyperparams_and_time()
        self.curr_steps_count += 1
        self.this_lm_total_batches += 1
        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def on_train_epoch_start(self):
        self.curr_steps_count = 0

    def log_time(self):
        if None is not self.start_time:
            self.log_stat("process_time", process_time() - self.start_time)

    def on_train_epoch_end(self):
        self.log_stat("train_loss", (sum(self.curr_train_losses) /
                                     len(self.curr_train_losses)))
        self.log_time()
        self.curr_train_losses = []

    def on_validation_epoch_end(self):
        val_loss = sum(self.curr_val_losses) / len(self.curr_val_losses)
        self.log_stat("validation_loss", val_loss)
        self.log_time()
        self.last_val_loss = val_loss  # might want this e.g. in model_explorer
        self.curr_val_losses = []
        if self.samples_at_validation:
            print("====\ncurrent val loss:", val_loss, ", sampling: ====")
            sample = self.lm.sample(
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

    def get_loss(self, batch, from_train=False):
        indices, mask = batch["x_indices"], batch["mask"]
        if mask is not None:
            indices = indices + (mask*self.lm.tokenizer.pad_token_id)
        x = indices[:, :-1]
        y = indices[:, 1:]  # -> y not contiguous
        # -> y.view(-1) won't work, need reshape instead
        y = y.to(dtype=torch.long)  # cross entropy loss expects target to have
        # type 'long' and will crash without explanation otherwise, so lets
        # just be safe
        z = self.lm(x)
        if from_train:
            self.n_train_samples += x.shape[0]
        return self.loss(z.view(-1, self.lm.n_tokens), y.reshape(-1))

    def curr_avg_lr(self):
        lrs = [pg['lr'] for pg in self.lr_schedulers().optimizer.param_groups]
        return -1 if not lrs else sum(lrs) / len(lrs)

    def log_hyperparams_and_time(self):
        n_active_params = sum(p.numel() for p in
                              self.lm.parameters() if p.requires_grad)
        self.log_stat("n_active_params", n_active_params)
        self.log_time()

    def maybe_log_hyperparams_and_time(self):
        freq = self.train_params.hyperparams_log_freq
        if self.curr_steps_count % freq == 0:
            self.log_hyperparams_and_time()

    def training_step(self, batch, batch_idx):
        # not actually a "training_step" rather, a step in the computation
        # of the loss for the optimizer step (which may be only every x
        # train_steps, specifically x=accumulate_grad_batches)
        loss = self.get_loss(batch, from_train=True)
        self.log("train_batch_loss", loss.item())  # for the lr scheduler
        self.curr_train_losses.append(loss.item())
        self.log_stat("train_batch_loss", loss.item())
        self.log_stat("avg_lr", self.curr_avg_lr())
        self.log_stat("n_train_samples", self.n_train_samples)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.curr_val_losses.append(loss.item())
        return loss.item()

    def make_main_scheduler(self, optimizer):
        if self.train_params.scheduler_type == 'Plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau
            return sched(optimizer, mode='min', verbose=False,
                         min_lr=self.train_params.min_lr,
                         factor=self.train_params.scheduler_factor,
                         patience=self.train_params.patience)
        elif self.train_params.scheduler_type == 'Cyclic':
            sched = torch.optim.lr_scheduler.CyclicLR
            return sched(optimizer, base_lr=self.train_params.min_lr,
                         max_lr=self.train_params.lr,
                         step_size_up=self.train_params.lr_cycle_steps // 2,
                         mode='triangular',
                         gamma=self.train_params.scheduler_factor,
                         cycle_momentum=False)
        elif self.train_params.scheduler_type == 'Cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR
            return sched(optimizer, self.train_params.lr_cycle_steps,
                         eta_min=self.train_params.min_lr)
        else:
            raise Exception("unknown scheduler type:",
                            self.train_params.scheduler_type)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.train_params.lr)

        def f_warmup(n):
            assert n <= self.train_params.warm_steps
            return n / self.train_params.warm_steps
        s_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, f_warmup)
        s_main = self.make_main_scheduler(optimizer)
        s_full = MyChainedScheduler(optimizer, [s_warmup, s_main],
                                    milestones=[self.train_params.warm_steps])
        # get scheduler started, else first batch has max value apparently
        s_full.step(None)
        s_main = {"scheduler": s_full,
                  "monitor": "train_batch_loss",
                  "interval": "step",
                  "reduce_on_plateau": True}
        # lightning wont pass the loss through if you dont tell it that
        # it's reduce_on_plateau >:(
        return [optimizer], [s_main]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class MyChainedScheduler:
    def __init__(self, optimizer, schedulers, milestones):
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
