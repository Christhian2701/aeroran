import csv
import random

# Configuration
output_filename = "5s_hierarchical_actions.csv"
duration_ms = 5000  # Total duration to generate actions for (1 second)
interval_ms = 100   # 100ms E2 periodicity
cells = [2, 3, 4, 5, 6, 7, 8]      # The IDs of our mmWave gNBs in the simple scenario

def generate_actions():
    actions = []
    # Start at timestamp 100ms, go up to 1000ms
    for timestamp in range(interval_ms, duration_ms + interval_ms, interval_ms):
        for cell_id in cells:
            # Action type is 0 for Energy Saving
            action_type = 0
            
            # Randomly pick 1 (ON) or 0 (OFF)
            # We heavily weight it towards ON (e.g., 70% chance) so we don't 
            # instantly crash the whole network, but still get some OFF events.
            ho_allowed = 1 if random.random() < 0.7 else 0 
            
            actions.append([timestamp, action_type, cell_id, ho_allowed])
            
    return actions

def main():
    actions = generate_actions()
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # The external control parser expects: timestamp, action_type, param1, param2
        # No headers! Just raw data.
        writer.writerows(actions)
        
    print(f"Generated {len(actions)} actions and saved to {output_filename}")
    
    # Print a preview so you can verify
    print("\nPreview of generated actions:")
    for row in actions[:10]:
        print(f"Timestamp: {row[0]}ms | Cell {row[2]} -> State {row[3]}")

if __name__ == "__main__":
    main()