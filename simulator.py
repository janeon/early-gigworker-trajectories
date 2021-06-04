# import torch
# from torch import nn
# from torch.distributions import Binomial, Poisson
from scipy.stats import binom, poisson
import numpy as np
import matplotlib.pyplot as plt
import pandas

# Given lambda (Poisson) for bids, p (binomial) offers per bid *as model parameters*
# Using PyTorch operations, sample number of bids, sample number of offers from each bid
# Up to here, this is how we would generate data, AND ALSO how we would write the beginning part of the model

# ... want to get "probability of k bids, j offers" out of model
#   can we get p(b bids, o offers | lambda, p) as a number for each b, o?
#   yes by mininizing for the Wasserstein distance
#   If not, do we basically need to sample it lots of times and then say our resulting "dataset" is our "distribution",
#   assuming that when you sample from this distribution you're just uniformly drawing from the dataset?
#   Potentially look at https://dfdazac.github.io/sinkhorn.html

def main():
    lamb = 3  #  on average, FLs drop off around lambda
    bids = 6  # maximal number of bids, minus 1
    p = 0.6 # global bid-winning probability 
    b_pmfs, drop_offs = initial_probabilities(bids, p, lamb)
    # print(drop_offs)
    cml_drop_offs, cml_pmfs = apply_drop_offs(b_pmfs, drop_offs, bids)
    final_probabilies = construct_probs_matrix(b_pmfs, cml_pmfs, cml_drop_offs, bids)
    printable = np.vstack([final_probabilies, drop_offs])
    # print(printable)
    row_labels = [str(i)+" offers" for i in range(bids)] + ["Drop off rate"]
    col_labels = [str(i)+ " bids" for i in range(bids)] 
    # print("\nPROBABILITIES WHERE ROWS=OFFERS & COLUMNS=BIDS")
    df = pandas.DataFrame(printable, columns=col_labels, index=row_labels)
     # round off to two places for viewing
    print(df.round(3), "\n")

def initial_probabilities(bids, p, lamb):
    # Constructing binomial probabilities without account for dropoffss
    b_pmfs =  [ [ binom.pmf(row, col, p) for col in range(bids) ] for row in range(bids) ]
    b_pmfs = np.array(b_pmfs)

    # using the poisson to extract expected dropoff rate at each bid
    p_x = np.arange(1, bids+1, 1)
    p_y = poisson.pmf(p_x, lamb)
    # plt.plot(p_x, p_y)
    # plt.show()
    drop_offs = [0]
    drop_offs.extend(np.asarray(p_y[:-1]))
    # for y in p_y[1:-1]:
    #     drop_offs.append(y*drop_offs[-1])
    # print("drop_offs", drop_offs)
    # print("b_pmfs", b_pmfs)
    return b_pmfs, drop_offs

def apply_drop_offs(b_pmfs, drop_offs, bids):
    # applying dropoff rates to each bid column
    scaled_pmfs = [drop_offs[col]*b_pmfs[:,col] for col in range(bids)]
    scaled_pmfs = np.column_stack(scaled_pmfs)
    # print("scaled_pmfs", scaled_pmfs)

    # finding column-wise cumulative sum for computing probabilities later
    cml_drop_offs = np.cumsum(np.array(drop_offs))
    cml_pmfs = np.cumsum(scaled_pmfs, axis=1)
    # print("cml_drop_offs", cml_drop_offs)
    # print("cml_pmfs", cml_pmfs)
    return cml_drop_offs, cml_pmfs

def construct_probs_matrix(b_pmfs, cml_pmfs, cml_drop_offs, bids):
    # constructing probabilities accounting for drop-off rates
    final_probabilies = b_pmfs[:,:2] # start off with first two columns
    # note that the first bid isn't scaled at all because here we don't account for
    # freelancers who drop off after 0th bid (those who register but don't bid at all)
    for i in range(1, bids-1):
        # constructing column i+1
        prev_column = final_probabilies[:,-1:]
        rate = cml_drop_offs[i] # cumulative dropoff rate up to prev col = rate for current col = # of FLs left
        prev_weight = 1-rate  # rate given to previous column
        base_column = b_pmfs[:,i+1:i+2] # i+1th column of the binomial matrix
        curr_column = prev_weight*prev_column  + rate*base_column # assign weights to both columns
        final_probabilies = np.column_stack((final_probabilies, curr_column))
    return final_probabilies

if __name__ == "__main__":
    main()

# m = Poisson(torch.tensor([4]))
# print(m.sample())

# x = torch.distributions.Binomial(total_count=100, probs=p)
# print(x.sample())

# y = torch.distributions.Binomial(torch.tensor([100]), probs=torch.tensor([p]))
# print(y.sample())

# z = torch.rand(4,4) * 5
# print(z)

# print(torch.bernoulli(a))
# rates = torch.rand(no_fls) * lamb # so suppose freelancers bid anywhere from 1 to 10 times a day
# fl_bids = torch.poisson(rates)

# m = Binomial(fl_bids, torch.Tensor(no_fls).fill_(p))
# x = m.sample()
# print("bids:", fl_bids)
# print("offers:", x)
