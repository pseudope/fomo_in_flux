import importlib
import json
import numpy as np
from omegaconf import ListConfig
import termcolor
import tqdm

import data_lib

def update_args(args):
    dataset_names = args.experiment.dataset.name
    has_dataset_names = dataset_names is not None and len(dataset_names)
    has_sequence = args.experiment.dataset.sequence is not None
    # Using dataset.name to use and combine specific full dataset sequences instead of given sequences.
    assert_str = "Both experiment.dataset.name and experiment.dataset.sequence are undefined. Please define either one."
    assert has_dataset_names or has_sequence, assert_str
    if not has_sequence:
        sequence = []
        for dataset_name in dataset_names:
            sequence.append(json.load(open(f'stream_construction/datasets/{dataset_name}.json', 'r')))
        sequence = [x for y in sequence for x in y]
    else:
        # Using a predefined sequence directly.
        sequence = json.load(open(args.experiment.dataset.sequence, 'r'))
    
    if args.experiment.dataset.sequence_reshuffle:
        termcolor.cprint('Using shuffled data sequence!\n', 'yellow', attrs=[])
        np.random.seed(args.experiment.seed)
        np.random.shuffle(sequence)
        
    assert_str = f'Using predefined sequence {args.experiment.dataset.sequence}, but number of classes provided is smaller than the number of tasks required ({args.experiment.task.num})!'
    assert len(sequence) >= args.experiment.task.num, assert_str
    
    assert_str = f'Currently only supporting sequences with unique class + dataset combinations!'
    assert len(sequence) == len(np.unique(sequence)), assert_str
    
    sequence = [x.split('+++') for x in sequence]
    # Convert ALL_CAPS_DATASET_NAME to dataset handle if needed.
    temp_sequence = []
    for x in sequence:
        if x[0] in data_lib.ConversionHandleData:
            temp_sequence.append(x)
        else:
            temp_sequence.append([data_lib.ConversionDataHandle[x[0]], x[1]])
    sequence = temp_sequence
    dataset_names = []
    for x in sequence:
        if x[0] not in dataset_names:
            dataset_names.append(x[0])

    # Add target info.
    # This is where we create the final data sequence structure that is passed everywhere:
    # sequence = updated_sequence = [entry1 = {}, ...] with 
    # entry1 = {
    #   handle: name of dataset, 
    #   classname: classname_str, 
    #   local_target: target value for dataset accounting for potential lack of total classes, 
    #   target: target value according to data_lib.<dataset>.BASE_CLASSES ordering,
    #   global_local_target: target value with respect to total number of provided classes,
    #   global_target: target value with respect to total number of available classes for each referenced dataset.
    # }
    
    updated_sequence = []
    classname2target = {}
    localclassname2target = {}        
    task_target_offsets = {}
    full_target_offsets = {}
    task_cumsum, full_cumsum = 0, 0            
    referenced_handle_classes = {}
    for handle, classname in sequence:
        if handle not in referenced_handle_classes:
            referenced_handle_classes[handle] = []
        if classname not in referenced_handle_classes[handle]:
            referenced_handle_classes[handle].append(classname)
            
    for handle, classname in tqdm.tqdm(sequence, desc='Preparing loading sequence...'):
        try:
            mod = importlib.import_module(f'data_lib.{handle}')
            if handle not in classname2target:
                # Backward compatibility due to naming mismatch in streams.
                # rev_bcc = lambda x: x
                # if handle == 'mvtecad_eval':
                #     rev_bcc = {value: key for key, value in mod.backward_classname_compatibility.items()}
                #     classname2target[handle] = {rev_bcc[clsn]: i for i, clsn in enumerate(mod.BASE_CLASSES)}
                # else:
                classname2target[handle] = {clsn: i for i, clsn in enumerate(mod.BASE_CLASSES)}

                # Compute relevant global offsets.
                task_target_offsets[handle] = task_cumsum
                task_cumsum += len(mod.BASE_CLASSES)
                
                # Compute relevant global local offsets.
                handle_classnames = [clsn for clsn in mod.BASE_CLASSES if clsn in referenced_handle_classes[handle]]
                localclassname2target[handle] = {clsn: i for i, clsn in enumerate(handle_classnames)}
                full_target_offsets[handle] = full_cumsum
                full_cumsum += len(localclassname2target[handle])
                
            sequence_info = {
                'dataset': handle,
                'classname': classname,
                'local_target': localclassname2target[handle][classname],
                'target': classname2target[handle][classname],
                'global_local_target': full_target_offsets[handle] + localclassname2target[handle][classname],
                'global_target': task_target_offsets[handle] + classname2target[handle][classname]
            }
        except Exception as e:
            print(e)
            from IPython import embed; embed()
            raise Exception(f"Error with dataset {handle}!")
        updated_sequence.append(sequence_info)

    if args.experiment.task.num_samples_per_task is not None:
        raise NotImplementedError('Currently only support fixed number of tasks.')

    args.experiment.dataset.name = dataset_names

    # We break down sequence  = [{dataset, ...} x num_entries ] into [[{dataset, ...}, ...], ...] x num_tasks].
    if args.experiment.task.dataset_incremental:
        sequence = []
        datasets_covered = []
        sub_sequence = []
        for entry in updated_sequence:
            if entry['dataset'] not in datasets_covered:
                if len(sub_sequence):
                    sequence.append(sub_sequence)
                sub_sequence = []
                datasets_covered.append(entry['dataset'])
            sub_sequence.append(entry)
        sequence.append(sub_sequence)        
    else:
        num_tasks = args.experiment.task.num
        classes_per_task = len(updated_sequence) // num_tasks
        overflow = len(updated_sequence) % num_tasks
        classes_per_task = [classes_per_task + 1 if i < overflow else classes_per_task for i in range(num_tasks)]
        cumsums = [0] + list(np.cumsum(classes_per_task))
        sequence = [updated_sequence[cumsums[i]:cumsums[i+1]] for i in range(num_tasks)]        

    args.experiment.dataset.sequence = sequence     
       
    # If dataset.name is not a list (i.e. using a single dataset), convert it to a list regardless.
    if not isinstance(args.experiment.dataset.name, ListConfig):
        args.experiment.dataset.name = [args.experiment.dataset.name]
        
    return sequence, dataset_names        