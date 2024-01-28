import arviz as az
import matplotlib.pyplot as plt

#1
centred_data = az.load_arviz_data("centered_eight")
non_centred_data = az.load_arviz_data("non_centered_eight")


num_chains_centered = centred_data.posterior.dims['chain'] # numarul de lanturi
sample_size_centered = centred_data.posterior.dims['draw'] # marimea esantionului

num_chains_non_centered = non_centred_data.posterior.dims['chain'] # numarul de lanturi
sample_size_non_centered = non_centred_data.posterior.dims['draw'] # marimea esantionului

print(f"Modelul centrat: Numărul de lanțuri = {num_chains_centered}, Mărimea eșantionului = {sample_size_centered}")
print(f"Modelul necentrat: Numărul de lanțuri = {num_chains_non_centered}, Mărimea eșantionului = {sample_size_non_centered}")

az.plot_posterior(centred_data) # afisarea posterioarelor
az.plot_posterior(non_centred_data) # afisarea posterioarelor

#2
# compararea cu R-hat
rhat_centered = az.rhat(centred_data, var_names=['mu', 'tau'])
rhat_non_centered = az.rhat(non_centred_data, var_names=['mu', 'tau'])
print("R-hat pentru modelul centrat:\n", rhat_centered)
print("R-hat pentru modelul necentrat:\n", rhat_non_centered)

# autocorelatia 
az.plot_autocorr(centred_data, var_names=['mu', 'tau'])
az.plot_autocorr(non_centred_data, var_names=['mu', 'tau'])

#3
num_divergences_centered = centred_data.sample_stats.diverging.sum()
num_divergences_non_centered = non_centred_data.sample_stats.diverging.sum()
print(f"Modelul centrat: Numărul de divergențe = {num_divergences_centered}")
print(f"Modelul necentrat: Numărul de divergențe = {num_divergences_non_centered}")

mu_tau_centered = centred_data.posterior[['mu', 'tau']] 
mu_tau_non_centered = non_centred_data.posterior[['mu', 'tau']] 

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5), constrained_layout=True)
data_list = [centred_data, non_centred_data]

for idx, tr in enumerate(data_list):
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter', divergences=True, divergences_kwargs={'color':'C1'}, ax=ax[idx])
    ax[idx].set_title(['centered', 'non-centered'][idx])

plt.show()

#sau

# #centered
# divergences_centered = centered_eight_data.sample_stats["diverging"].sum()
# print(f"Numarul de divergente pentru modelul centrat: {divergences_centered}")
# az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
# plt.suptitle("Model Centrat- divergente")
# plt.show()

# #non-centered
# divergences_non_centered = non_centered_eight_data.sample_stats["diverging"].sum()
# print(f"Numarul de divergente pentru modelul necentrat: {divergences_non_centered}")
# az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)
# plt.suptitle("Model Necentrat- divergente")
# plt.show()