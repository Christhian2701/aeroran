import sem

# primeira tentativa de campanha, usando cenário simples

ns_path = '/home/christhian/iwcmc_oran'
#script_path = '/home/christhian/iwcmc_oran/scratch/scenario-simple-test.cc'
#script_path = 'scenario-simple-test'
# /home/christhian/iwcmc_oran/scratch/scenario-hierarchical-xangai-UAV.cc

script_path = 'scenario-hierarchical-xangai-UAV'

campaign_path = '/home/christhian/iwcmc_oran/0teste'


control_paths =[
    "/home/christhian/iwcmc_oran/1_offline_train/ahierarchical_actions.csv",
    "/home/christhian/iwcmc_oran/1_offline_train/bhierarchical_actions.csv",
    "/home/christhian/iwcmc_oran/1_offline_train/chierarchical_actions.csv",
    "/home/christhian/iwcmc_oran/1_offline_train/dhierarchical_actions.csv",
]


campaign = sem.CampaignManager.new(ns_path, script_path, campaign_path, overwrite=True, check_repo=False, max_parallel_processes=4)

params = {
    'simTime': '30.0',
    'RngRun': list(range(2)),
    'controlFileName': control_paths,
    'uavMobilityMode': 1,
    'uavFlightPattern': 1,
    'positionAllocator': 2,
    'enableTraces': 'true',
    'pathGymOkMetrics': 'true',
    'useSemaphores': 'false',
    'scheduleControlMessages': 'false',
}

all_simulations = sem.utils.list_param_combinations(params)

print(f"\nTotal simulations to run: {len(all_simulations)}\n")

#print(campaign)

for sim in all_simulations:
    print(sim)

campaign.run_simulations(all_simulations)
