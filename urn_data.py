from chancedataset import urns
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os

SEED = 64

def main():
    parser = argparse.ArgumentParser(description='Create some urn data')
    parser.add_argument('filename', help='the file name to use')
    parser.add_argument('n_urns', help='the number of urns to be constructed', type=int)
    parser.add_argument('n_probe', help='the number of probe samples to be produced per urns', type=int)
    parser.add_argument('n_chance', help='the number of chance samples to be produced per urns', type=int)
    parser.add_argument('n_likely', help='the number of likely samples to be produced per urns', type=int)
    parser.add_argument('n_between', help='the number of between samples to be produced per urns', type=int)
    parser.add_argument('prompt', help='add the prompt and instructions', type=bool)

    args = parser.parse_args()

    colors = np.asarray(['red', 'blue', 'green', 'black', 'white']).reshape(5,1)
    max = 25
    min = 1

    rng = np.random.default_rng(SEED)

    # Build the urns
    new_urns = []
    for i in range(1, colors.shape[0]):
        c = colors[:i+1]
        for j in range(args.n_urns):
            balls = rng.integers(low=min, high=max, size=i+1).reshape(i+1,1)
            u = np.hstack((c, balls)) 
            new_urns.append(u)
    
    # Create the urn generators and build the dataset
    flag = True
    for u in tqdm(new_urns):
        gen = urns.UrnGenerator(u, rng)
        ns = [args.n_probe, args.n_chance, args.n_likely, args.n_between]
        tasks = [i for i in range(4)]
        data = gen.generate_samples(ns, tasks, prompt=args.prompt, unique=True)
        if flag:
            total_data = data
            flag = False
        else:
            total_data = np.vstack((total_data, data))
    
    # Conver the Data to Pandas
    print('Converting to csv and saving.')
    df = pd.DataFrame(total_data, columns=['Question', 'Answer'])

    path = os.getcwd()
    os.makedirs('TestData', exist_ok=True)
    df.to_csv('TestData/' + args.filename)

if __name__ == "__main__":
    main()