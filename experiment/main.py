# Standard Library Modules
import time
import argparse
# Custom Modules
from utils.arguments import ArgParser
from utils.utils import set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random seed
    if args.seed not in [None, 'None']:
        set_random_seed(args.seed)

    start_time = time.time()

    # Get the job to do
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'captioning':
            if args.job == 'preprocessing':
                from task.captioning.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.captioning.train import training as job
            elif args.job == 'testing':
                if args.task_dataset in ['flickr8k', 'flickr30k']:
                    from task.captioning.flickr_test import testing as job
                elif args.task_dataset in ['coco2014', 'coco2017']:
                    from task.captioning.coco_test import testing as job
                elif args.task_dataset == 'uit_viic':
                    from task.captioning.uit_test import testing as job
                elif args.task_dataset == 'aide':
                    from task.captioning.aide_test import testing as job
                elif args.task_dataset == 'new_lv':
                    from task.captioning.lv_et_test import testing as job
                elif args.task_dataset == 'new_et':
                    from task.captioning.lv_et_test import testing as job
                elif args.task_dataset == 'new_fi':
                    from task.captioning.lv_et_test import testing as job
            elif args.job == 'eval_similarity':
                from task.captioning.eval_similarity import eval_similarity as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'annotating':
            if args.job == 'gpt_annotating':
                if args.task_dataset in ['flickr8k', 'flickr30k', 'coco2014']:
                    from task.annotating.gpt_annotating_multiprocess_ko import gpt_annotating_multiprocess_ko as job
                elif args.task_dataset == 'uit_viic':
                    from task.annotating.gpt_annotating_multiprocess_vie import gpt_annotating_multiprocess_vie as job
                elif args.task_dataset == 'aide':
                    from task.annotating.gpt_annotating_multiprocess_pl import gpt_annotating_multiprocess_pl as job
                elif args.task_dataset == 'new_lv':
                    from task.annotating.gpt_annotating_multiprocess_lv import gpt_annotating_multiprocess_lv as job
                elif args.task_dataset == 'new_et':
                    from task.annotating.gpt_annotating_multiprocess_et import gpt_annotating_multiprocess_et as job
                elif args.task_dataset == 'new_fi':
                    from task.annotating.gpt_annotating_multiprocess_fi import gpt_annotating_multiprocess_fi as job
            elif args.job == 'backtrans_annotating':
                from task.annotating.backtrans_annotating_easynmt import backtrans_annotating as job
            elif args.job == 'eda_annotating':
                from task.annotating.eda_annotating import eda_annotating as job
            elif args.job == 'synonym_annotating':
                from task.annotating.synonym_annotating import synonym_annotating as job
            elif args.job == 'onlyone_annotating':
                from task.annotating.onlyone_annotating import onlyone_annotating as job
            elif args.job == 'budget_annotating':
                from task.annotating.budget_annotating import budget_annotating as job
            elif args.job == 'translation_annotating':
                if args.task_dataset == 'uit_viic':
                    from task.annotating.translation_annotating_vie import translation_annotating_vie as job
                elif args.task_dataset == 'aide':
                    from task.annotating.translation_annotating_pl import translation_annotating_pl as job
                elif args.task_dataset == 'new_lv':
                    from task.annotating.translation_annotating_new import translation_annotating_lv as job
                elif args.task_dataset == 'new_et':
                    from task.annotating.translation_annotating_new import translation_annotating_et as job
                elif args.task_dataset == 'new_fi':
                    from task.annotating.translation_annotating_new import translation_annotating_fi as job
            elif args.job == 'googletrans_annotating':
                from task.annotating.googletrans_annotating import googletrans_annotating as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'text_style_transfer':
            if args.job == 'preprocessing':
                from task.text_style_transfer.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.text_style_transfer.train import training as job
            elif args.job == 'testing':
                from task.text_style_transfer.test import testing as job
        elif args.task == 'annotating_tst':
            if args.job == 'gpt_annotating':
                from task.annotating_tst.gpt_annotating import gpt_annotating_multiprocess as job
            elif args.job == 'translation_annotating':
                from task.annotating_tst.translation_annotating import translation_annotating as job
            elif args.job == 'googletrans_annotating':
                from task.annotating_tst.googletrans_annotating import googletrans_annotating as job
        elif args.task == 'style_classification':
            if args.job == 'preprocessing':
                from task.style_classification.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.style_classification.train import training as job
            elif args.job == 'testing':
                from task.style_classification.test import testing as job
            elif args.job == 'inference':
                from task.style_classification.inference import inference as job
        elif args.task == 'machine_translation':
            if args.job == 'preprocessing':
                from task.machine_translation.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.machine_translation.train import training as job
            elif args.job == 'testing':
                from task.machine_translation.test import testing as job
        elif args.task == 'annotating_mt':
            if args.job == 'gpt_annotating':
                from task.annotating_mt.gpt_annotating import gpt_annotating_multiprocess as job
            elif args.job == 'translation_annotating':
                from task.annotating_mt.translation_annotating import translation_annotating as job
            elif args.job == 'googletrans_annotating':
                from task.annotating_mt.googletrans_annotating import googletrans_annotating as job
        else:
            raise ValueError(f'Invalid task: {args.task}')

    # Do the job
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    # Run the main function
    main(args)
