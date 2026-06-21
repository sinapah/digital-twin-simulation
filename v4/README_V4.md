# V4 Federated Learning with Multipass VMs

## Overview

V4 implements a federated learning system using **Multipass VMs** where:
- **3 Edge VMs** receive images from UA-DETRAC intersections, train locally, and participate in federated aggregation
- **1 Aggregator VM** coordinates federated learning by receiving model weights from edge VMs and broadcasting updated global weights
- All VMs communicate over **TCP** for weight exchange
- Each edge VM handles **8 intersections** (24 intersections ÷ 3 edges = 8 per edge)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UA-DETRAC (24 intersections)                    │
│                    MVI-01, MVI-02, ..., MVI-24 (sorted)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Edge VM 0     │       │   Edge VM 1     │       │   Edge VM 2     │
│                 │       │                 │       │                 │
│ Intersections:  │       │ Intersections:  │       │ Intersections:  │
│ 0-7 (8 total)   │       │ 8-15 (8 total)  │       │ 16-23 (8 total) │
│                 │       │                 │       │                 │
│ Python sender   │       │ Python sender   │       │ Python sender   │
│ process streams │       │ process streams │       │ process streams │
│ images from its │       │ images from its │       │ images from its │
│ assigned 8      │       │ assigned 8      │       │ assigned 8      │
│ intersections   │       │ intersections   │       │ intersections   │
│                 │       │                 │       │                 │
│ Local training  │       │ Local training  │       │ Local training  │
│ with TCP upload │       │ with TCP upload │       │ with TCP upload │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                         ┌───────────────────────┐
                         │   Aggregator VM       │
                         │                       │
                         │ Receives weights from │
                         │ all 3 edge VMs via    │
                         │ TCP                   │
                         │                       │
                         │ Performs FedAvg       │
                         │ aggregation           │
                         │                       │
                         │ Broadcasts updated    │
                         │ global weights to all │
                         │ edge VMs via TCP      │
                         └───────────────────────┘
```

## VM Setup (Multipass)

### Prerequisites

```bash
# Install Multipass (Ubuntu)
sudo snap install multipass

# Verify installation
multipass version
```

### Launch 4 VMs

Run these commands to create your 4 VMs:

```bash
# Create edge VM 0
multipass launch --name edge-vm-0 --cpus 2 --memory 4G --disk 20G

# Create edge VM 1
multipass launch --name edge-vm-1 --cpus 2 --memory 4G --disk 20G

# Create edge VM 2
multipass launch --name edge-vm-2 --cpus 2 --memory 4G --disk 20G

# Create aggregator VM
multipass launch --name aggregator-vm --cpus 2 --memory 4G --disk 20G
```

### Get VM IP Addresses

```bash
# Get all VM IPs
multipass list

# Or get specific VM IP
multipass info edge-vm-0 | grep "IPv4"
multipass info edge-vm-1 | grep "IPv4"
multipass info edge-vm-2 | grep "IPv4"
multipass info aggregator-vm | grep "IPv4"
```

### Set Up Each VM

Run these commands on **each VM** (edge-vm-0, edge-vm-1, edge-vm-2, aggregator-vm):

```bash
# SSH into the VM
multipass shell edge-vm-0  # or edge-vm-1, edge-vm-2, aggregator-vm

# Update and install dependencies
sudo apt update
sudo apt install -y python3 python3-pip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install torch torchvision matplotlib pandas pillow psutil

# Exit VM
exit
```

### Copy V4 Code to Each VM

```bash
# Copy v4 directory to each VM
multipass transfer /path/to/v4 edge-vm-0:/home/ubuntu/
multipass transfer /path/to/v4 edge-vm-1:/home/ubuntu/
multipass transfer /path/to/v4 edge-vm-2:/home/ubuntu/
multipass transfer /path/to/v4 aggregator-vm:/home/ubuntu/

# Copy UA-DETRAC data to edge VMs only (not needed on aggregator)
multipass transfer /path/to/DETRAC-Images edge-vm-0:/home/ubuntu/
multipass transfer /path/to/DETRAC-Images edge-vm-1:/home/ubuntu/
multipass transfer /path/to/DETRAC-Images edge-vm-2:/home/ubuntu/

multipass transfer /path/to/DETRAC-Train-Annotations-XML edge-vm-0:/home/ubuntu/
multipass transfer /path/to/DETRAC-Train-Annotations-XML edge-vm-1:/home/ubuntu/
multipass transfer /path/to/DETRAC-Train-Annotations-XML edge-vm-2:/home/ubuntu/
```

## Running the System

### Step 1: Start the Aggregator

On the **aggregator VM**:

```bash
multipass shell aggregator-vm
cd ~/v4
source venv/bin/activate

# Start the aggregator (default port 5000)
python3 aggregator.py --port 5000
```

The aggregator will:
- Listen for incoming connections from edge VMs
- Wait for all 3 edges to connect
- Perform federated aggregation (FedAvg)
- Broadcast updated weights to all edges

### Step 2: Start Edge VMs

On **each edge VM** (in parallel or sequentially):

```bash
# Edge VM 0
multipass shell edge-vm-0
cd ~/v4
source venv/bin/activate

# Start edge agent 0 (intersections 0-7)
python3 edge_agent.py \
  --edge-id 0 \
  --intersection-indices 0 1 2 3 4 5 6 7 \
  --aggregator-host <aggregator-vm-ip> \
  --aggregator-port 5000 \
  --sender-port 6000
```

```bash
# Edge VM 1
multipass shell edge-vm-1
cd ~/v4
source venv/bin/activate

# Start edge agent 1 (intersections 8-15)
python3 edge_agent.py \
  --edge-id 1 \
  --intersection-indices 8 9 10 11 12 13 14 15 \
  --aggregator-host <aggregator-vm-ip> \
  --aggregator-port 5000 \
  --sender-port 6001
```

```bash
# Edge VM 2
multipass shell edge-vm-2
cd ~/v4
source venv/bin/activate

# Start edge agent 2 (intersections 16-23)
python3 edge_agent.py \
  --edge-id 2 \
  --intersection-indices 16 17 18 19 20 21 22 23 \
  --aggregator-host <aggregator-vm-ip> \
  --aggregator-port 5000 \
  --sender-port 6002
```

### Step 3: Start Senders on Each Edge VM

On **each edge VM**, start the sender process that streams images from UA-DETRAC:

```bash
# Edge VM 0 - Start sender for intersections 0-7
python3 sender.py \
  --edge-id 0 \
  --intersection-indices 0 1 2 3 4 5 6 7 \
  --target-host <edge-vm-0-ip> \
  --target-port 6000 \
  --fps 25 \
  --max-frames-per-video 50
```

```bash
# Edge VM 1 - Start sender for intersections 8-15
python3 sender.py \
  --edge-id 1 \
  --intersection-indices 8 9 10 11 12 13 14 15 \
  --target-host <edge-vm-1-ip> \
  --target-port 6001 \
  --fps 25 \
  --max-frames-per-video 50
```

```bash
# Edge VM 2 - Start sender for intersections 16-23
python3 sender.py \
  --edge-id 2 \
  --intersection-indices 16 17 18 19 20 21 22 23 \
  --target-host <edge-vm-2-ip> \
  --target-port 6002 \
  --fps 25 \
  --max-frames-per-video 50
```

### Quick Start Script

Create a `start_all.sh` script on your host machine:

```bash
#!/bin/bash

AGGREGATOR_IP="192.168.64.2"  # Replace with actual aggregator VM IP
EDGE0_IP="192.168.64.3"       # Replace with actual edge-vm-0 IP
EDGE1_IP="192.168.64.4"       # Replace with actual edge-vm-1 IP
EDGE2_IP="192.168.64.5"       # Replace with actual edge-vm-2 IP

echo "Starting aggregator..."
multipass exec aggregator-vm -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 aggregator.py --port 5000
" &

sleep 2

echo "Starting edge agents..."
multipass exec edge-vm-0 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 edge_agent.py --edge-id 0 --intersection-indices 0 1 2 3 4 5 6 7 --aggregator-host $AGGREGATOR_IP --aggregator-port 5000 --sender-port 6000
" &

multipass exec edge-vm-1 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 edge_agent.py --edge-id 1 --intersection-indices 8 9 10 11 12 13 14 15 --aggregator-host $AGGREGATOR_IP --aggregator-port 5000 --sender-port 6001
" &

multipass exec edge-vm-2 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 edge_agent.py --edge-id 2 --intersection-indices 16 17 18 19 20 21 22 23 --aggregator-host $AGGREGATOR_IP --aggregator-port 5000 --sender-port 6002
" &

sleep 2

echo "Starting senders..."
multipass exec edge-vm-0 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 sender.py --edge-id 0 --intersection-indices 0 1 2 3 4 5 6 7 --target-host $EDGE0_IP --target-port 6000 --fps 25 --max-frames-per-video 50
"

multipass exec edge-vm-1 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 sender.py --edge-id 1 --intersection-indices 8 9 10 11 12 13 14 15 --target-host $EDGE1_IP --target-port 6001 --fps 25 --max-frames-per-video 50
"

multipass exec edge-vm-2 -- bash -c "
  cd ~/v4 && source venv/bin/activate
  python3 sender.py --edge-id 2 --intersection-indices 16 17 18 19 20 21 22 23 --target-host $EDGE2_IP --target-port 6002 --fps 25 --max-frames-per-video 50
"
```

## TCP Communication Protocol

### Aggregator → Edge Connection

- **Port**: Configurable (default: 5000)
- **Protocol**: TCP sockets
- **Message types**:
  - `WEIGHTS_UPDATE`: Broadcast updated global weights to all edges
  - `ROUND_START`: Signal edges to begin training round
  - `ROUND_COMPLETE`: Confirm round completion

### Edge → Aggregator Connection

- **Port**: Same as aggregator listening port (default: 5000)
- **Protocol**: TCP sockets
- **Message types**:
  - `CONNECT`: Initial connection handshake with edge ID
  - `WEIGHTS_UPLOAD`: Send local model weights to aggregator
  - `READY`: Signal that edge is ready for next round
  - `STATUS`: Send training metrics (loss, accuracy, samples processed)

### Edge → Sender Connection

- **Port**: Per-edge configurable (default: 6000, 6001, 6002)
- **Protocol**: TCP sockets
- **Message types**:
  - `IMAGE_BATCH`: Send batch of images from UA-DETRAC
  - `LABELS`: Send corresponding labels
  - `DONE`: Signal end of stream for an intersection

## File Structure

```
v4/
├── aggregator.py          # Federated learning aggregator (runs on aggregator VM)
├── edge_agent.py          # Edge device agent (runs on each edge VM)
├── sender.py              # UA-DETRAC image sender (runs on each edge VM)
├── utils/
│   ├── tcp_comm.py        # TCP communication utilities
│   ├── detrac_loader.py   # UA-DETRAC dataset loader
│   └── metrics.py         # Training metrics collection
├── requirements.txt
└── README_V4.md          # This file
```

## Key Design Decisions

### Why TCP instead of UDP?

1. **Reliability**: Weight updates must arrive intact; TCP guarantees delivery
2. **Ordering**: Federated aggregation requires consistent weight ordering
3. **Flow control**: TCP's congestion control prevents overwhelming receivers
4. **Simplicity**: No need to implement custom reliability layer

### Why 8 intersections per edge?

- 24 intersections ÷ 3 edges = 8 intersections per edge
- Each edge handles a manageable subset
- Balanced workload distribution
- Allows for studying intersection-level variations

### Multipass VMs

- **Isolation**: Each VM runs independently
- **Reproducibility**: VM state can be snapshotted and restored
- **Resource control**: CPU/memory/disk limits per VM
- **Network control**: Easy to configure VM-to-VM networking

## Troubleshooting

### VM Connectivity Issues

```bash
# Check if VMs can reach each other
multipass exec edge-vm-0 -- ping <aggregator-vm-ip>

# Check firewall rules
multipass exec aggregator-vm -- sudo ufw status
```

### Port Already in Use

```bash
# Change port in edge_agent.py and sender.py
# Default: aggregator port 5000, sender ports 6000-6002
```

### Connection Refused

```bash
# Ensure aggregator is running first
# Check aggregator is listening: netstat -tlnp | grep 5000
# Verify IP addresses: multipass list
```

## Training Configuration & Performance

### Round Structure
Each federated learning round consists of:
1. **Local Training**: Each edge trains on its assigned data
2. **Weight Exchange**: Edges send weights to aggregator → Aggregator performs FedAvg → Aggregator sends updated weights to edges
3. **Metrics Collection**: Performance and resource metrics are reported

### Images Per Round
Each edge processes:
- **Per intersection**: Up to 50 frames per video (configurable via `--max-frames-per-video`)
- **Per edge**: 8 intersections × 50 frames = 400 frames maximum
- **Batch size**: 64 images per batch
- **Actual processed**: 
  - With UA-DETRAC: Up to 400 frames × 8 intersections = 3,200 frames per edge per round
  - With dummy data (testing): Exactly 100 samples per round

With the current configuration:
- Each edge completes ~2 weight update steps per epoch (100 samples ÷ 64 batch size)
- Total training steps per round: 3 epochs × 2 steps = 6 steps

### Round Duration
A typical round takes approximately:
- **Local Training**: `LOCAL_EPOCHS` epochs over the dataset
- **Weight Transfer**: Time to serialize/transfer model weights (~5-50MB depending on model size)
- **Aggregation**: Time for FedAvg computation (negligible for small models)
- **Network Latency**: VM-to-VM communication time

With the current configuration (`LOCAL_EPOCHS=3`, `BATCH_SIZE=64`, dummy dataset of 100 samples):
- Each edge completes ~2 weight update steps per epoch (100 samples ÷ 64 batch size)
- Total training steps per round: 3 epochs × 2 steps = 6 steps
- Typical round time: 5-15 seconds on local machine, longer over network

### Epochs & Batching
- **LOCAL_EPOCHS**: 3 (number of times edge trains on its local data per round)
- **BATCH_SIZE**: 64 (images processed together during training)
- **Learning Rate**: 1e-3 (Adam optimizer)

### Total Training Rounds
- **DEFAULT_ROUNDS**: 100 (configurable in aggregator.py)
- **Total optimization steps**: 100 rounds × 3 epochs × ~2 steps = ~600 steps
- **Total images processed**: 100 rounds × 100 samples = 10,000 samples (with dummy data)

### Queueing Behavior
The system implements flow control at multiple levels:

1. **TCP Socket Buffers**: Automatic flow control prevents overwhelming receivers
2. **Message Size Prefixing**: All messages include 8-byte size prefix so receivers know exactly how much data to expect
3. **Separate Channels**: 
   - Control messages (JSON): Small metadata (~100 bytes)
   - Weight data (Binary): Large payloads sent via `torch.save()` (much more compact than JSON)
4. **Aggregator Threading**: Dedicated message-handling threads per edge prevent blocking

### Resource Monitoring
CPU usage is monitored during local training:
- Samples taken every 0.01 seconds during training batches
- Reports `cpu_avg` and `cpu_peak` per round per edge
- Values are percentages of a single CPU core (can exceed 100% on multi-core systems)

## Next Steps

1. Create the V4 directory structure
2. Implement `aggregator.py` with FedAvg
3. Implement `edge_agent.py` with local training and TCP upload
4. Implement `sender.py` to stream UA-DETRAC images
5. Implement `utils/tcp_comm.py` for reliable message passing
6. Test with 1 round, then scale up
7. Collect metrics and analyze convergence
