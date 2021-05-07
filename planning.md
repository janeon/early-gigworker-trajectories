# Planning prototype

- Write the simulation as a differentiable program
- ## Come up with training objective for our model to minimize
- Sample random parameters, use these to generate data according to simulation
- Use optimizer to minimize training objective on simulated data, to find "learned parameters"
- Compare learned parameters to the true parameters used to generate the data

Milestones:

- Create the simplest possible Pytorch program, that only looks at people bidding repeatedly (drawn according to a Poisson w/ some $\lambda$), w/ all freelancers have the same winning probability p, and getting offers according to a Binomial process
  - Go through the above steps for this simple program
  - Learn how to create a differentiable sample from a probability distribution
  - Come up with objective to minimize (KL divergence? of parameter p)
- Add complexity in stages
  - Add "prior" component, "quitting" component/threshold, replaces Poisson in terms of generating number of bids
    - Output is the total number of bids and offers you got before you quit
    - We'll use a Beta prior, and have the initial "starting" prior values be learned
  - Add distribution over initial priors, so that different people quit at different times based on the same evidence
  - Add distribution over probabilities p of actually winning an offer from a bid
