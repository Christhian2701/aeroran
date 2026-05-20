import sem

# primeira tentativa de campanha, usando cenário simples

ns_path = '.'
#script_path = 'scenario-simple-test'

script_path = 'scenario-hierarchical-xangai-UAV'

campaign_path = './new_run1'


control_paths =[
   #f"{ns_path}/1_offline_train/ehierarchical_actions.csv",
    f"{ns_path}/1_offline_train/ahierarchical_actions.csv",
    f"{ns_path}/1_offline_train/bhierarchical_actions.csv",
    f"{ns_path}/1_offline_train/chierarchical_actions.csv",
    f"{ns_path}/1_offline_train/dhierarchical_actions.csv",
    f"{ns_path}/1_offline_train/ehierarchical_actions.csv",
    f"{ns_path}/1_offline_train/fhierarchical_actions.csv",
    f"{ns_path}/1_offline_train/ghierarchical_actions.csv",
    f"{ns_path}/1_offline_train/hhierarchical_actions.csv",
]


campaign = sem.CampaignManager.new(ns_path, script_path, campaign_path, overwrite=True, check_repo=False, max_parallel_processes=4)

params = {
    'simTime': '30.0',
    'RngRun': list(range(1)),
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
