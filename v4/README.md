# =========================================================
# V4 - Federated Learning with Multipass VMs
# =========================================================

This directory contains the V4 implementation of a federated learning system using Multipass VMs.

## Architecture

- **3 Edge VMs**: Each trains on 8 UA-DETRAC intersections (24 total ÷ 3 = 8 per edge)
- **1 Aggregator VM**: Coordinates federated learning using FedAvg
- **Communication**: TCP sockets for weight exchange

## Quick Start (Local Testing)

```bash
cd v4
./setup.sh
./quickstart.sh
```

This starts all components on your local machine for testing.

## Production (Multipass VMs)

### 1. Launch 4 VMs

```bash
multipass launch --name edge-vm-0 --cpus 2 --memory 4G --disk 20G
multipass launch --name edge-vm-1 --cpus 2 --memory 4G --disk 20G
multipass launch --name edge-vm-2 --cpus 2 --memory 4G --disk 20G
multipass launch --name aggregator-vm --cpus 2 --memory 4G --disk 20G
```

### 2. Setup Each VM

```bash
# Get VM IPs
multipass list

# Setup each VM
for vm in edge-vm-0 edge-vm-1 edge-vm-2 aggregator-vm; do
    multipass transfer v4 $vm:/home/ubuntu/
    multipass transfer DETRAC-Images $vm:/home/ubuntu/  # Skip for aggregator
    multipass transfer DETRAC-Train-Annotations-XML $vm:/home/ubuntu/  # Skip for aggregator
    
    multipass exec $vm -- bash -c "
        cd ~/v4 && ./setup.sh
    "
done
```

### 3. Start the System

**On Aggregator VM:**
```bash
multipass shell aggregator-vm
cd ~/v4
source venv/bin/activate
python3 aggregator.py --port 5000
```

**On Edge VMs (parallel):**
```bash
# Edge 0
multipass shell edge-vm-0
cd ~/v4
source venv/bin/activate
python3 edge_agent.py \
    --edge-id 0 \
    --intersection-indices 0 1 2 3 4 5 6 7 \
    --aggregator-host <aggregator-ip> \
    --aggregator-port 5000 \
    --sender-port 6000

# Edge 1
multipass shell edge-vm-1
cd ~/v4
source venv/bin/activate
python3 edge_agent.py \
    --edge-id 1 \
    --intersection-indices 8 9 10 11 12 13 14 15 \
    --aggregator-host <aggregator-ip> \
    --aggregator-port 5000 \
    --sender-port 6001

# Edge 2
multipass shell edge-vm-2
cd ~/v4
source venv/bin/activate
python3 edge_agent.py \
    --edge-id 2 \
    --intersection-indices 16 17 18 19 20 21 22 23 \
    --aggregator-host <aggregator-ip> \
    --aggregator-port 5000 \
    --sender-port 6002
```

**On Edge VMs (senders - parallel):**
```bash
# Start senders on each edge VM
python3 sender.py --edge-id 0 --intersection-indices 0 1 2 3 4 5 6 7 --target-host <edge0-ip> --target-port 6000
python3 sender.py --edge-id 1 --intersection-indices 8 9 10 11 12 13 14 15 --target-host <edge1-ip> --target-port 6001
python3 sender.py --edge-id 2 --intersection-indices 16 17 18 19 20 21 22 23 --target-host <edge2-ip> --target-port 6002
```

## File Structure

```
v4/
├── aggregator.py          # Federated learning aggregator
├── edge_agent.py          # Edge device agent
├── sender.py              # UA-DETRAC image sender
├── utils/
│   ├── tcp_comm.py        # TCP communication utilities
│   ├── detrac_loader.py   # UA-DETRAC dataset loader
│   └── metrics.py         # Training metrics collection
├── setup.sh               # Setup script for VMs
├── run.sh                 # Run script
├── quickstart.sh          # Quick start for local testing
├── requirements.txt       # Python dependencies
└── README_V4.md          # This file
```

## Key Components

### Aggregator (`aggregator.py`)
- Listens on TCP port for edge connections
- Performs FedAvg aggregation
- Broadcasts updated weights to all edges
- Coordinates federated learning rounds

### Edge Agent (`edge_agent.py`)
- Receives images from sender
- Trains local model
- Uploads weights to aggregator
- Downloads updated weights
- Participates in federated learning

### Sender (`sender.py`)
- Streams images from UA-DETRAC
- Sends to edge agent via TCP
- Rate-limited to match FPS

### Utilities (`utils/`)
- `tcp_comm.py`: TCP communication helpers
- `detrac_loader.py`: UA-DETRAC data loading
- `metrics.py`: Training metrics collection

## Configuration

Edit these values in the Python files:
- `DEFAULT_PORT`: Aggregator listening port
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `LOCAL_EPOCHS`: Local training epochs per round
- `IMG_SIZE`: Image resize target

## Troubleshooting

### Connection Refused
- Ensure aggregator is running first
- Verify IP addresses: `multipass list`
- Check firewall: `sudo ufw status`

### Port Already in Use
- Change port in the Python files
- Kill existing processes: `pkill -f aggregator.py`

### Slow Training
- Reduce `LOCAL_EPOCHS`
- Reduce `max_frames_per_video` in sender
- Use GPU: `torch.device('cuda')`

## Next Steps

1. Implement full UA-DETRAC data loading in `detrac_loader.py`
2. Implement image serialization in `tcp_comm.py`
3. Add model checkpointing and recovery
4. Implement real-time metrics visualization
5. Add outage simulation and fallback handling
