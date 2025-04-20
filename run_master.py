#!/usr/bin/env python
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the MasterNode class
from src.master.node import MasterNode

# Initialize the master node
master = MasterNode(host='0.0.0.0', port=65432)
master.start()

# Initialize a small model (for demonstration)
print('Initializing and sharding model...')
model_path = master.initialize_model('facebook/opt-125m', num_shards=2)
print(f'Model initialized and sharded at: {model_path}')

# Wait for worker connections
print('\nWaiting for worker connections...')
print('Please start worker nodes in separate terminals with:')
print('python run_worker.py')

# Keep the script running
try:
    while True:
        cmd = input("\nEnter command (assign/distribute/inference/exit): ")
        if cmd == 'assign':
            assignments = master.assign_shards()
            print(f'Shard assignments: {assignments}')
        elif cmd == 'distribute':
            master.distribute_shards()
            print('Shards distributed to workers')
        elif cmd == 'inference':
            text = input('Enter text for inference: ')
            result = master.run_inference(text)
            print(f'Inference result: {result}')
        elif cmd == 'exit':
            break
        else:
            print('Unknown command')
except KeyboardInterrupt:
    pass
finally:
    # Stop the master node
    master.stop()
    print('Master node stopped')