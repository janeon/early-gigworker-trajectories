import torch
from torch import nn
from torch.distributions import Binomial, Poisson

# m = Poisson(torch.tensor([4]))
# print(m.sample())
lamb = 5
no_fls = 10
p = 0.3

# Given lambda (Poisson) for bids, p (binomial) offers per bid *as model parameters*
# Using PyTorch operations, sample number of bids, sample number of offers from each bid
# Up to here, this is how we would generate data, AND ALSO how we would write the beginning part of the model
rates = torch.rand(no_fls) * lamb # so suppose freelancers bid anywhere from 1 to 10 times a day
fl_bids = torch.poisson(rates)

m = Binomial(fl_bids, torch.Tensor(no_fls).fill_(p))
x = m.sample()
print("bids:", fl_bids)
print("offers:", x)

# ... want to get "probability of k bids, j offers" out of model
#   can we get p(b bids, o offers | lambda, p) as a number for each b, o?
# yes by mininizing for the Wasserstein distance
#   If not, do we basically need to sample it lots of times and then say our resulting "dataset" is our "distribution",
#   assuming that when you sample from this distribution you're just uniformly drawing from the dataset?
#   Potentially look at https://dfdazac.github.io/sinkhorn.html

# hmmm, mininizing the Wasserstein distance finds the closest distance betwene two probability distributions? feels like we only have one from the simulation so far