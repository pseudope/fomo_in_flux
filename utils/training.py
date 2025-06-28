# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import termcolor
import torch
import tqdm
import wandb

import continual_lib
import data_lib
import experiment_lib
import utils.conf
import utils.evaluation

# for memory utils
import psutil

plt.switch_backend("agg")

# fix the buggy itertools cycle which leads to memory leaks
# this fix is taken from: https://github.com/pytorch/pytorch/issues/23900
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            
def setup_wandb(args: omegaconf.DictConfig, wandb_run_id=None) -> None:
    """Setup Weights & Biases logging."""

    termcolor.cprint("Setting up Weights & Biases logging.", "blue", attrs=[])
    Path(args.log.folder).mkdir(parents=True, exist_ok=True)
    if args.log.wandb_key is not None:
        if "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_API_KEY"] = args.log.wandb_key
        else:
            assert (
                os.environ["WANDB_API_KEY"] == args.log.wandb_key
            ), "WANDB_API_KEY is set and different from the one in the config file"

    run_folder = os.path.join(args.log.folder, 'wandb')
    os.makedirs(run_folder, exist_ok=True)
    name = args.log.name
    if name is None:
        name = args.log.id
    resume = wandb_run_id is not None

    wandb.init(
        project=args.log.project,
        group=args.log.group,
        name=name,
        dir=run_folder,
        settings=wandb.Settings(start_method="fork"),
        resume=resume,
        id=wandb_run_id
    )
    run_id = wandb.run.id
    args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.config.update(args)
    return run_id

def train(
    args: omegaconf.DictConfig,
    continual_learner: continual_lib.BaseContinualLearner,
    experiment: experiment_lib.PredefinedSequenceExperiment,
    evaluator: utils.evaluation.Evaluator
) -> None:
    """
    """
    global_counter = 0
    starting_task = 0
    training_stats = []

    # If checkpointing is turned on, will search for possible existing checkpoint to resume from there.
    wandb_run_id = None
    if args.log.checkpoint and os.path.exists(
        args.log.log_folder / "checkpoint.pth.tar"
    ):
        termcolor.cprint(
            f'\nContinuing from checkpoint {args.log.log_folder/"checkpoint.pth.tar"}.\n',
            "green",
            attrs=["bold"],
        )
        chkpt_file = torch.load(args.log.log_folder / "checkpoint.pth.tar")
        continual_learner.load_from_checkpoint(chkpt_file["continual_learner"])
        experiment.load_from_checkpoint(chkpt_file["experiment"])
        evaluator.load_from_checkpoint(chkpt_file["evaluator"])
        wandb_run_id = chkpt_file["wandb.run.id"]
        starting_task = chkpt_file["task"]
        global_counter = chkpt_file["global_counter"]

    #Set up Weights & Biases if needed.
    if args.log.use:
        wandb_run_id = setup_wandb(args, wandb_run_id)

    if hasattr(continual_learner, 'begin_training'):
        continual_learner.begin_training(experiment)
    
    # Store initial models for CLIP score computation
    initial_backbone = copy.deepcopy(continual_learner.backbone.module).eval()
    initial_text_encoder = None
    initial_tokenizer = None
    if hasattr(continual_learner.head.module, 'text_encoder'):
        initial_text_encoder = copy.deepcopy(continual_learner.head.module.text_encoder).eval()
        initial_tokenizer = continual_learner.head.module.tokenizer
    
    # print("initial_backbone:\n",initial_backbone)
    # print("initial_text_encoder:\n", initial_text_encoder)
    # print("initial_tokenizer:\n", initial_tokenizer)
    # raise ValueError("Initial models are set up, but training is not implemented yet. Please implement the training loop.")

    # Measure initial zero-shot performance. For non-pretrained models, should be ~ 1/#classes.
    if evaluator.pre_train_results is None:
        zero_start_time = time.time()
        termcolor.cprint(
            "Evaluating initial zero-shot performance on full evaluation data.\n",
            "green",
            attrs=[],
        )
        default_eval_result_dict = pre_train_result_dict = (
            evaluator.evaluate_and_summarize(
                continual_learner, experiment, subset_only=1
            )
        )

        subset_val = args.experiment.evaluation.validate_on_subset
        if subset_val < 1:
            termcolor.cprint(
                "Evaluating initial zero-shot performance on evaluation data subset.\n",
                "green",
                attrs=[],
            )
            pre_train_result_dict = evaluator.evaluate_and_summarize(
                continual_learner, experiment, subset_only=subset_val
            )

        # Beyond separately storing the pre-training results, we also include them in the training logging process
        # to easily visualize changes from the default performance.
        evaluator.assign_pre_train_results(pre_train_result_dict)

        # Log zero_shot results via W&B if needed.
        if args.log.use:
            # This logs the zeroshot performance data separately.
            default_log_dict = evaluator.prepare_log_dict(
                custom_log_data=default_eval_result_dict,
                custom_preemble="start_zeroshot",
                merge_to_train_results=False,
            )
            default_log_dict["global_counter"] = global_counter
            # Store initial zero-shot performance in local file.
            json.dump(
                default_log_dict,
                open(
                    os.path.join(evaluator.log_folder, "zeroshot_score_initial.json"),
                    "w",
                ),
            )
            # This logs the zeroshot performance (with optional subsetting) as part of the training log process.
            pre_train_log_dict = evaluator.prepare_log_dict(use_pre_train_results=True)
            # Upload to W&B.
            wandb.log(default_log_dict | pre_train_log_dict)
        print("Completed in {0:4.2f}s".format(time.time() - zero_start_time))

    # If we only perform zeroshot performance evaluation, no additional steps are required.
    if args.zeroshot_only:
        import sys
        sys.exit()


    # Start training on data stream.
    termcolor.cprint(
        f'\nBeginning {args.experiment.type.replace("_", "-")} data streaming.',
        "green",
        attrs=[],
    )
    task_iterator = range(starting_task, args.experiment.task.num)

    ################################################
    datacomp_loader = None
    for task in task_iterator:
        task_start_time = time.time()
        utils.conf.set_random_seed(task * 100000, set_backend=args.log.full_replication)

        # In case we use a buffer, we have to account for the fact that
        # during the first task, the buffer is empty, and we have to increase update 
        # (and optionally pretraining) batch_size.
        if args.experiment.task.update_pool_batch_size != args.experiment.task.batch_size:
            if task == 0:
                div = 1 + int(args.experiment.task.pretraining_batch_size > 0)
                update_base_batch_size = experiment.task_buffer_batch_size // div
                update_base_batch_size += experiment.task_buffer_batch_size % div
                experiment.task_batch_size += update_base_batch_size
                if div > 1:
                    update_pretrain_batch_size = experiment.task_buffer_batch_size // div
                    ### Grab DataComp dataloader if possible
                    datacomp_loader = None
                    if args.experiment.task.pretraining_batch_size > 0:
                    #     termcolor.cprint("\n> Setting up first-task DataComp Loader", "white", attrs=["bold"])
                    #     assert args.experiment.dataset.pretraining_data_path is not None, 'The data-mixture suggests to use pretraining-data in the training mixture, but args.experiment.dataset.pretraining_data_path is not set!'
                    #     datacomp_loader = data_lib.get_datacomp_loader(
                    #         args,
                    #         train_transform=args.experiment.dataset.train_transforms,
                    #         train_batch_size=args.experiment.task.pretraining_batch_size + update_pretrain_batch_size,
                    #         custom_seed=task * 100000 + args.experiment.seed + 1
                    #     ) if args.experiment.dataset.pretraining_data_path is not None else None
                    # datacomp_loader = cycle(datacomp_loader)
                        if args.experiment.dataset.pretraining_data_path is not None:
                            datacomp_root = args.experiment.dataset.pretraining_data_path
                            tar_files = [x for x in os.listdir(datacomp_root) if x.endswith('.tar')]
                            if len(tar_files) > 0:
                                termcolor.cprint("\n> Setting up first-task DataComp Loader", "white", attrs=["bold"])
                                datacomp_loader = data_lib.get_datacomp_loader(
                                    args,
                                    train_transform=args.experiment.dataset.train_transforms,
                                    train_batch_size=args.experiment.task.pretraining_batch_size + update_pretrain_batch_size,
                                    custom_seed=task * 100000 + args.experiment.seed + 1
                                )
                            else:
                                termcolor.cprint(
                                    f"Warning: no DataComp tar files found in {datacomp_root}. Skipping pretraining data.",
                                    "yellow"
                                )
                        else:
                            termcolor.cprint(
                                "Warning: pretraining batch size > 0 but no pretraining data path is set. Skipping pretraining data.",
                                "yellow"
                            )
                    if datacomp_loader is not None:
                        datacomp_loader = cycle(datacomp_loader)
                                        
            elif task == 1:  # potentially just set to >= 1 for full replication.
                experiment.task_batch_size = args.experiment.task.update_pool_batch_size
                if args.experiment.task.pretraining_batch_size > 0:
                    # termcolor.cprint("\n> Setting up final DataComp Loader", "white", attrs=["bold"])
                    # datacomp_loader = data_lib.get_datacomp_loader(
                    #     args,
                    #     train_transform=args.experiment.dataset.train_transforms,
                    #     train_batch_size=args.experiment.task.pretraining_batch_size,
                    #     custom_seed=task * 100000 + args.experiment.seed + 1
                    # ) if args.experiment.dataset.pretraining_data_path is not None else None
                    # datacomp_loader = cycle(datacomp_loader)
                    if args.experiment.dataset.pretraining_data_path is not None:
                        datacomp_root = args.experiment.dataset.pretraining_data_path
                        tar_files = [x for x in os.listdir(datacomp_root) if x.endswith('.tar')]
                        if len(tar_files) > 0:
                            termcolor.cprint("\n> Setting up final DataComp Loader", "white", attrs=["bold"])
                            datacomp_loader = data_lib.get_datacomp_loader(
                                args,
                                train_transform=args.experiment.dataset.train_transforms,
                                train_batch_size=args.experiment.task.pretraining_batch_size,
                                custom_seed=task * 100000 + args.experiment.seed + 1
                            )
                            datacomp_loader = cycle(datacomp_loader)
                        else:
                            termcolor.cprint(
                                f"Warning: no DataComp tar files found in {datacomp_root}. Skipping pretraining data.",
                                "yellow"
                            )
                    else:
                        termcolor.cprint(
                            "Warning: pretraining batch size > 0 but no pretraining data path is set. Skipping pretraining data.",
                            "yellow"
                        )
                
        # Get task-specific dataloaders.
        utils.conf.set_random_seed(task * 100000, set_backend=args.log.full_replication)
        task_dataloaders = experiment.give_task_dataloaders()
        train_loader = task_dataloaders['train']
        buffer_loader = task_dataloaders['buffer']

        # Collect statistics about the current task data composition
        task_stats = {
            'task': task,
            'num_new_data_samples': experiment.task_num_samples_train,
            'num_buffer_samples': experiment.task_num_samples_buffer,
        }
        if experiment.task_num_samples_buffer > 0:
            ratio = experiment.task_num_samples_train / experiment.task_num_samples_buffer
            task_stats['new_to_buffer_ratio'] = ratio
        else:
            task_stats['new_to_buffer_ratio'] = None

        num_datasets = sum([experiment.all_train_idcs[dataset_name][task] is not None for dataset_name in experiment.dataset_names])
        termcolor.cprint(
            f'\n----- Task {task+1}/{args.experiment.task.num} | Data from {num_datasets} dataset(s).', 'white', attrs=["bold"])
        if train_loader is None:
            break

        # If we bound the number of iterations (converted from number of samples seen) instead of the number of epochs,
        # we dynamically assign the number of epochs for each task.
        num_train_samples = args.experiment.task.n_samples
        task_stats['data_limit'] = num_train_samples

        train_iterations = num_train_samples // args.experiment.task.batch_size
        warmup_iterations = (
            args.experiment.task.n_warmup_samples // args.experiment.task.batch_size
        )
        milestone_iterations = [
            x // args.experiment.task.batch_size
            for x in args.experiment.scheduler.multistep_milestones
        ]
        stability_gap_iterations = [int(np.ceil(x / args.experiment.task.batch_size)) for x in args.experiment.task.gap_samples]
        steps_per_epoch = len(experiment.current_task_train_dataset)
        virtual_epochs = int(
            np.ceil(args.experiment.task.n_samples / steps_per_epoch)
        )
        # Update task metric collection
        base_log_context = {
            'task_num': task
        }

        # Set Up Continual Learner.
        continual_learner.train()
        continual_learner.prepare_for_training(
            experiment=experiment
        )
        continual_learner.begin_task(
            scheduler_steps=train_iterations, 
            milestone_steps=milestone_iterations)

        ################################################
        # Starting Continual Learning process.
        iter_count, is_unfrozen = 0, True

        if args.experiment.task.n_warmup_samples > 0:
            utils.conf.freeze_all_but_head(continual_learner)
            is_unfrozen = False

        # Adapt the number of iterations we require from the task dataloader.
        # This allows us to simple cycle through a data-loader even if 
        # num_train_samples is larger than the actual data-loader capacity.
        # However, we only artificially set the length if num_train_samples is larger
        # to avoid losing samples.
        assert_str = f'{train_loader.dataset} requires set_len() function!'
        assert hasattr(train_loader.dataset, 'set_len'), assert_str
        if num_train_samples > len(train_loader.dataset):
            train_loader.dataset.set_len(num_train_samples)
        if buffer_loader and num_train_samples > len(buffer_loader.dataset):
            buffer_loader.dataset.set_len(num_train_samples)
        
        utils.conf.set_random_seed(task * 100000, set_backend=args.log.full_replication)

        training_step_data = []
        seen_new_samples = 0
        seen_buffer_samples = 0
        with tqdm.tqdm(
            total=num_train_samples, position=0, desc="Training...", leave=True
        ) as pbar:
            #### Uncomment for printing training throughput
            # num_samples = 0
            # st = time.time()

            # Start Training.
            datacomp_iterator = datacomp_loader or [None] * len(train_loader)
            buffer_iterator = buffer_loader or [None] * len(train_loader)
            for batch_index, (data, buffer_batch, datacomp_batch) in enumerate(zip(train_loader, buffer_iterator, datacomp_iterator)):
                # batch_size = len(data['targets']) if args.experiment.task.data_mixture['update'] != 0 else 0

                # # if buffer_batch exists, couple it with base batch for training.
                # if buffer_batch is not None:
                #     batch_size += buffer_batch['images'].shape[0]

                # # if datacomp_batch exists, couple both batches together for training
                # if datacomp_batch is not None:
                #     batch_size += datacomp_batch[0].shape[0]

                update_size = len(data['targets']) if args.experiment.task.data_mixture['update'] != 0 else 0
                buffer_size = buffer_batch['images'].shape[0] if buffer_batch is not None else 0
                datacomp_size = datacomp_batch[0].shape[0] if datacomp_batch is not None else 0
                batch_size = update_size + buffer_size + datacomp_size
                seen_new_samples += update_size + datacomp_size
                seen_buffer_samples += buffer_size
                
                #### Uncomment for printing training throughput
                # tt = time.time() - st
                # num_samples += batch_size
                # if batch_index % 10 == 0:
                #     print('Throughput: {}'.format(num_samples / tt))
                    
                # (Optional) Warmup Functionality
                warmup_str = ''
                if args.experiment.task.n_warmup_samples > 0:
                    if iter_count < warmup_iterations:
                        warmup_str = '(WARMUP) '
                        continual_learner.freeze_features = True
                    else:
                        if not is_unfrozen:
                            continual_learner.freeze_features = False
                            utils.conf.unfreeze_all(continual_learner)
                            warmup_str, is_unfrozen = '', True

                # Print Statues.
                used_lrs = [x['lr'] for x in continual_learner.opt.param_groups]
                used_lrs_str = ['{0}'.format(x) for x in used_lrs]
                if len(used_lrs_str) <= 3:
                    used_lrs_str = ' | '.join(used_lrs_str)
                else:
                    used_lrs_str = ' | '.join(used_lrs_str[:3]) + ' (...)'
                epoch = (iter_count * batch_size) // steps_per_epoch + 1
                pbar.set_description('{0} Epoch {1}/{2}, LR = {3}. Samples seen'.format(
                    warmup_str, epoch, virtual_epochs, used_lrs_str))

                data = {
                    key: (
                        item.to(experiment.device)
                        if isinstance(item, torch.Tensor)
                        else item
                    )
                    for key, item in data.items()
                }

                ### Handle the case where there is no update data here:
                ### If there is a buffer batch, we replace the current data-items with the buffer batch
                ### If there is a buffer batch and a datacomp batch, we add to the current data since we already replaced the data batch
                ### If there is only a datacomp batch (no buffer batch), we replace the current data-items with the datacomp batch

                ## if buffer_batch exists add it to data
                if buffer_batch:
                    if args.experiment.task.data_mixture['update'] != 0:
                        data['images'] = torch.cat([data['images'], buffer_batch['images'].to(data['images'].device)], dim=0)
                        data['texts'] = data['texts'] + buffer_batch['texts']
                    else:
                        data['images'] = torch.cat([buffer_batch['images'].to(data['images'].device)], dim=0)
                        data['texts'] = buffer_batch['texts']

                ## if datacomp batch exists add it to data
                if datacomp_batch:
                    if args.experiment.task.data_mixture['update'] != 0:
                        data['images'] = torch.cat([data['images'], datacomp_batch[0].to(data['images'].device)], dim=0)
                        data['texts'] = data['texts'] + datacomp_batch[1]
                    else:
                        if buffer_batch:
                            data['images'] = torch.cat([data['images'], datacomp_batch[0].to(data['images'].device)], dim=0)
                            data['texts'] = data['texts'] + datacomp_batch[1]
                        else:
                            data['images'] = torch.cat([datacomp_batch[0].to(data['images'].device)], dim=0)
                            data['texts'] = datacomp_batch[1]
                
                # [CL] Compute base CL step.
                loss = continual_learner.observe(experiment=experiment, **data)
                if np.isinf(loss) or np.isnan(loss):
                    import sys
                    termcolor.cprint('Loss Infinite or NaN - exiting...\n', color='red', attrs=['bold'])
                    sys.exit()

                # [CL] Call optional End-Observation Functionality.
                continual_learner.end_observe()

                # Update Progressbar and local/global Counter Variables
                iter_count += 1
                global_counter += 1
                pbar.update(batch_size)

                # [CL] Compute Evaluation Metrics for Stability Gap Evaluations if needed.
                stability_gap_log_dict = {}
                if iter_count in stability_gap_iterations:
                    pbar.set_description("[Evaluating for Stability Gap Tests...]")
                    stability_gap_results_dict = evaluator.evaluate(
                        continual_learner,
                        experiment,
                        subset_only=args.experiment.evaluation.validate_on_subset,
                        exclude_from_aggregation=True,
                    )
                    stability_gap_log_dict = evaluator.prepare_log_dict(
                        stability_gap_results_dict
                    )
                    
                # Log data to W&B if needed.
                base_log_context['global_counter'] = global_counter
                if continual_learner.ON_TASK_LEARNING:
                    pbar.set_postfix_str('Loss: {0:3.5}'.format(loss))

                    if args.log.use:
                        aux_data = {}
                        aux_data['loss'] = loss
                        aux_data['image_mean'] = data["images"].cpu().numpy().mean()
                        for lr_idx, used_lr in enumerate(used_lrs):
                            aux_data[f'lr_group-{lr_idx}'] = used_lr
                        aux_data.update(stability_gap_log_dict)
                        log_data = base_log_context | aux_data
                        if not args.log.checkpoint:
                            wandb.log(log_data)
                        else:
                            training_step_data.append(log_data)
                else:
                    pbar.set_postfix_str('Training skipped - only collecting data.'.format(loss))

                if global_counter % train_iterations == 0:
                    break

        continual_learner.backbone.output_subset = None

        # [CL] Call Optional End-of-Task Functionality.
        eval_params = {
            'past_task_results': evaluator.results,
            'subset': 'model.end_task'
        }

        # sometimes we return some measurements from end task to track on wandb
        end_task_outs = continual_learner.end_task(experiment=experiment, eval_params=eval_params)

        # Move continual_learner into evaluation mode.
        continual_learner.prepare_for_evaluation(
            experiment=experiment,
            eval_params=eval_params
        )

        # Evaluate model on all encountered tasks.
        if continual_learner.ON_TASK_LEARNING:
            evaluate_now = True
            # Turn of evaluation if it is supposed to be performed every n-th task.
            if args.experiment.evaluation.every_nth_task > 0:
                if (task+1) % args.experiment.evaluation.every_nth_task != 0:
                    evaluate_now = False
            # Always evaluate on first and last task!
            if task == 0 or task == len(task_iterator)-1:
                evaluate_now = True
            if evaluate_now:
                _ = evaluator.evaluate(
                    continual_learner,
                    experiment,
                    subset_only=args.experiment.evaluation.validate_on_subset,
                )
                
            # After each task, if not set otherwise, a complete checkpoint is stored:
            if args.log.checkpoint and not args.log.checkpoint_no_recovery:
                chkpt_file = {
                    'task': task+1,
                    'global_counter': global_counter,
                    'continual_learner': continual_learner.checkpoint,
                    'experiment': experiment.checkpoint,
                    'evaluator': evaluator.checkpoint,
                    'wandb.run.id': wandb_run_id
                }
                ckpt_path = args.log.log_folder/'checkpoint.pth.tar'
                termcolor.cprint(f"Stored running checkpoint at {ckpt_path}.", "yellow", attrs=[])
                torch.save(chkpt_file, ckpt_path)
                if args.log.checkpoint_each_task:
                    task_ckpt_path = args.log.log_folder/f'checkpoint_{task+1}.pth.tar'
                    termcolor.cprint(f"Stored task checkpoint at {task_ckpt_path}.", "yellow", attrs=[])
                    torch.save(chkpt_file, task_ckpt_path)

            #Write evaluation results to JSON.
            evaluator.write_results(format='json')

            #If needed, sync data to Weight and Biases (on a task level).
            if args.log.use and evaluate_now:
                if args.log.checkpoint:
                    for train_step in training_step_data:
                        wandb.log(train_step)
                wandb_log_dict = evaluator.prepare_log_dict()
                wandb_log_dict['global_counter'] = global_counter
                # add other measurements from end-task
                if end_task_outs is not None:
                    wandb.log({**wandb_log_dict, **end_task_outs})
                else:
                    wandb.log(wandb_log_dict)

        task_end_time = time.time()
        if continual_learner.ON_TASK_LEARNING:
            print(
                "Task completed in {0:4.4f}s.\n".format(
                    task_end_time - task_start_time
                )
            )
        else:
            if args.continual.method != "joint":
                print(
                    "No training - buffer collection completed in {0:4.4f}s.".format(
                        task_end_time - task_start_time
                    )
                )
                
        # Save statistics for current task
        task_stats['seen_new_samples'] = seen_new_samples
        task_stats['seen_buffer_samples'] = seen_buffer_samples
        task_stats['total_seen_samples'] = seen_new_samples + seen_buffer_samples
        training_stats.append(task_stats)

        # Compute CLIP scores for this task if filtering is enabled
        clip_ratio = args.experiment.buffer.clip_filter_ratio
        clip_mode = getattr(args.experiment.buffer, 'clip_filter_mode', 'random')
        if clip_mode != 'random': # clip_ratio < 1 and 
            if clip_mode == 'updated':
                b_model = continual_learner.backbone.module
                t_encoder = continual_learner.head.module.text_encoder
                tokenizer = continual_learner.head.module.tokenizer
            else:
                b_model = initial_backbone
                t_encoder = initial_text_encoder
                tokenizer = initial_tokenizer
            experiment.register_clip_scores_for_task(task, b_model, t_encoder, tokenizer)

            # Compute statistics and log CLIP scores
            clip_means = {}
            for ds in experiment.dataset_names:
                idcs = experiment.all_train_idcs[ds][task]
                if idcs is None or len(idcs) == 0:
                    continue
                scores = experiment.clip_scores[ds][idcs]
                valid = ~np.isnan(scores)
                if np.any(valid):
                    clip_means[ds] = float(np.mean(scores[valid]))
            task_stats['clip_score_means'] = clip_means
            experiment.dump_clip_scores_for_task(task, evaluator.log_folder)
        
        # Update task information in experiment.
        experiment.finish_task()

    # IF experiment.evaluation.additional_datasets_subets < 1, evaluation is only performed on a subset of the data during training after each task.
    # This means it is important that after having seen all tasks a full and complete evaluation run is performed.
    final_zero_start_time = time.time()
    termcolor.cprint(
        "\nEvaluating final zero-shot performance on full evaluation data.\n", 
        "green", 
        attrs=[])
    final_eval_result_dict = evaluator.evaluate_and_summarize(
        continual_learner, experiment, subset_only=1)
    if args.log.use:
        # This logs the final zeroshot performance data separately.
        final_log_dict = evaluator.prepare_log_dict(
            custom_log_data=final_eval_result_dict, 
            custom_preemble="start_zeroshot")
        # Store final results in json file.
        json.dump(
            final_log_dict, 
            open(
                os.path.join(
                    evaluator.log_folder, 
                    'zeroshot_score_final.json'), 'w'))
        # Upload to W&B.
        wandb.log(final_log_dict)
    print('Completed in {0:4.2f}s'.format(time.time() - final_zero_start_time))

    # Store training statistics
    stats_file = os.path.join(evaluator.log_folder, 'training_stats.json')
    json.dump(training_stats, open(stats_file, 'w'), indent=4)
    termcolor.cprint(f"Training statistics written to {stats_file}", "cyan")
    for stats in training_stats:
        termcolor.cprint(json.dumps(stats, indent=2), "cyan")
