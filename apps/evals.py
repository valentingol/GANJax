import argparse 
import pickle

from utils.evals import frechet_inception_distance


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, help='Evaluation method')
    parser.add_argument('--dataset_path', type=str, help='Real dataset on which to compute the evaluation metric')
    parser.add_argument('--generated_path', type=str, help='Generated dataset on which to compute the evaluation metric')
    args = parser.parse_args()

    with open(args.generated_path, 'rb') as generated:
        images = pickle.load(generated)
    

    if args.eval == 'FID':
        print('Evaluating FID score...')
        assert 'dataset_path' in args
        images_eval = (images + 1)/2 # rescale values between 0 and 1 for evaluation

        score = frechet_inception_distance(images_eval, args.dataset_path, img_size=(256,256))
        print('FID score : ' + str(score))