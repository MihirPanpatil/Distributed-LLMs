#!/usr/bin/env python
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the WorkerNode class
from src.worker.node import WorkerNode

# Initialize the worker node
worker = WorkerNode(master_address="localhost:65432")
print(f"Worker node initialized and connecting to master at localhost:65432")

# Start the worker
try:
    worker.start()
except KeyboardInterrupt:
    pass
finally:
    # Stop the worker node
    worker.stop()
    print('Worker node stopped')