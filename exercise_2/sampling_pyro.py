from torch.distributions.multivariate_normal import MultivariateNormal as TorchMVN
import torch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC


#note: running parallel chains is broken in some jupyter notebook environments. just run the scripts in your terminal.
def sample_likelihood(log_likelihood_fn, init_samples, warmup_steps, num_samples, step_size):
    
    def potential_fn(params): #the potential functions used by NUTS/HMC are the negative log-likelihood. 
        return -log_likelihood_fn(params['x']) #parameters need to be dicts
    num_chains = init_samples.shape[0]
    mcmc_kernel = NUTS(
        potential_fn=potential_fn, #cannot be a lambda function when using parallel chains
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False, #sometimes this is unstable
        step_size = step_size
    )
    mcmc = MCMC(
        mcmc_kernel,
        initial_params={'x':init_samples}, #parameters need a name
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        
    )
    mcmc.run()
    samples = mcmc.get_samples(group_by_chain=False)['x']
    #todo: need arviz
    return samples

#define sampling from a 5D normal distribution
D=5
L = torch.randn(D,D,dtype=torch.float64)
C=L@L.T
normal_dist = TorchMVN(torch.zeros(D,dtype=torch.float64), covariance_matrix = C)

#we want to use 4 parallel chains so we crease 4 initial samples
init_samples = torch.randn(4,D,dtype=torch.float64)
samples = sample_likelihood(normal_dist.log_prob, init_samples, 500, 500, 0.1)


#print the mean and covariance estimates
print("mean est.", torch.mean(samples,axis=0).detach().numpy())
print("cov est.", torch.cov(samples.T).detach().numpy())
print("cov truth.", C.detach().numpy())

#now some plotting
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('agg')
#plot the first two dimensions as scatter plot
plt.figure()
plt.scatter(samples[:,0],samples[:,1],label="samples")
plt.xlabel("$x_0$")
plt.xlabel("$x_1$")
plt.legend()
plt.savefig("samples_test.png")
plt.close()


