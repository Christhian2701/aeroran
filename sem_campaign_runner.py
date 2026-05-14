import sem

# primeira tentativa de campanha, usando cenário simples

ns_path = '/home/christhian/iwcmc_oran/'
#script_path = '/home/christhian/iwcmc_oran/scratch/scenario-simple-test.cc'
#script_path = 'scenario-simple-test'
# /home/christhian/iwcmc_oran/scratch/scenario-hierarchical-xangai-UAV.cc

script_path = 'scenario-hierarchical-xangai-UAV'

campaign_path = '/home/christhian/iwcmc_oran/0teste'

campaign = sem.CampaignManager.new(ns_path, script_path, campaign_path, overwrite=True, check_repo=False, max_parallel_processes=4)

params = {
    'simTime': '0.2',
    'RngRun': list(range(3)),
    'controlFileName':"none"
}

all_simulations = sem.utils.list_param_combinations(params)

print(f"Total simulations to run: {len(all_simulations)}")

print(campaign)
campaign.run_simulations(all_simulations)