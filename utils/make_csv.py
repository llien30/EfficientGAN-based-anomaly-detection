import argparse
import glob
import pandas as pd
import os

def get_arguments():
    parser = argparse.ArgumentParser(description='make csv files')

    parser.add_argument(
        '--dataset_dir',
        type = str,
        default = './data',
        help = 'input the PATH of the directry where the datasets are saved')

    parser.add_argument(
        '--save_dir',
        type = str,
        default = './csv',
        help = 'input the PATH of the directry where the csv files will be saved')

    return parser.parse_args()

def main():
    args = get_arguments()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for split in ['train', 'test']:
        img_paths = []
        cls_ids = []
        cls_labels = []

        class2ids = {'NORMAL':0, 'ABNORMAL':1}

        for c in ['NORMAL', 'ABNORMAL']:
            img_dir = os.path.join(args.dataset_dir, split, c)
            paths = glob.glob(os.path.join(img_dir,'*.jpg'))

            img_paths += paths
            cls_ids += [class2ids[c] for _ in range(len(paths))]
            cls_labels += [c for _ in range(len(paths))]

        df = pd.DataFrame({
            'img_path':img_paths,
            'cls_id':cls_ids,
            'cls_label':cls_labels},
            columns=['img_path','cls_id','cls_label'])

        df.to_csv(os.path.join(args.save_dir,'{}.csv').format(split), index=None)

    print('Done')

if __name__ == '__main__':
    main()