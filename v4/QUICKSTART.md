# V4: Federated Learning with Multipass VMs

## Summary

I've created V4, a federated learning system with **4 Multipass VMs** (3 edge + 1 aggregator) that communicate over **TCP**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              UA-DETRAC (24 intersections)                   │
│              MVI-01, MVI-02, ..., MVI-24                    │
└─────────────────────────────────────────────────────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Edge VM 0    │  │ Edge VM 1    │  │ Edge VM 2    │
│              │  │              │  │              │
│ Intersections│  │ Intersections│  │ Intersections│
│ 0-7 (8)      │  │ 8-15 (8)     │  │ 16-23 (8)    │
│              │  │              │  │              │
│ Local train  │  │ Local train  │  │ Local train  │
│ TCP upload   │  │ TCP upload   │  │ TCP upload   │
└───────┬──────┘  └───────┬──────┘  └───────┬──────┘
        └─────────────────┼─────────────────┘
                          │
                          ▼
                  ┌──────────────────┐
                  │ Aggregator VM    │
                  │                  │
                  │ FedAvg agg       │
                  │ Broadcast weights│
                  └──────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `v4/README.md` | Quick start guide |
| `v4/README_V4.md` | Detailed architecture and setup |
| `v4/aggregator.py` | Federated learning coordinator (runs on aggregator VM) |
| `v4/edge_agent.py` | Edge device with local training (runs on each edge VM) |
| `v4/sender.py` | UA-DETRAC image streamer (runs on each edge VM) |
| `v4/utils/tcp_comm.py` | TCP communication utilities |
| `v4/utils/detrac_loader.py` | UA-DETRAC dataset loader |
| `v4/utils/metrics.py` | Training metrics collection |
| `v4/setup.sh` | VM setup script |
| `v4/run.sh` | Component runner script |
| `v4/quickstart.sh` | Local testing script |
| `v4/requirements.txt` | Python dependencies |

## How to Run

### 1. Launch 4 Multipass VMs

```bash
multipass launch --name edge-vm-0 --cpus 2 --memory 4G --disk 20G
multipass launch --name edge-vm-1 --cpus 2 --memory 4G --disk 20G
multipass launch --name edge-vm-2 --cpus 2 --memory 4G --disk 20G
multipass launch --name aggregator-vm --cpus 2 --memory 4G --disk 20G
```

### 2. Setup VMs

```bash
# Get IPs
multipass list

# Setup each VM
for vm in edge-vm-0 edge-vm-1 edge-vm-2 aggregator-vm; do
    multipass transfer v4 $vm:/home/ubuntu/
    multipass exec $vm -- bash -c "cd ~/v4 && ./setup.sh"
done
```

### 3. Start Components

**Aggregator VM:**
```bash
multipass shell aggregator-vm
cd ~/v4 && source venv/bin/activate
python3 aggregator.py --port 5000
```

**Edge VMs (parallel):**
```bash
# Edge 0
python3 edge_agent.py \
  --edge-id 0 \
  --intersection-indices 0 1 2 3 4 5 6 7 \
  --aggregator-host <aggregator-ip> \
  --aggregator-port 5000 \
  --sender-port 6000

# Edge 1
python3 edge_agent.py \
  --edge-id 1 \
  --intersection-indices 8 9 10 11 12 13 14 15 \
  --aggregator-host <aggregator-ip> \
  --aggregator-port 5000 \
  --sender-port 6001

# Edge 2
python3 edge_agent.py \
  --edge-id 2 \
  --intersection-indices 16 17 18 19 20 21 22 23 \
  --aggregator-host <aggregator-ip> \
  --aggregator-port 5000 \
  --sender-port 6002
```

**Senders (parallel):**
```bash
# Start senders on each edge VM
python3 sender.py --edge-id 0 --intersection-indices 0 1 2 3 4 5 6 7 --target-host <edge0-ip> --target-port 6000
python3 sender.py --edge-id 1 --intersection-indices 8 9 10 11 12 13 14 15 --target-host <edge1-ip> --target-port 6001
python3 sender.py --edge-id 2 --intersection-indices 16 17 18 19 20 21 22 23 --target-host <edge2-ip> --target-port 6002
```

## Key Features

- ✅ **3 Edge VMs** with 8 intersections each (24 total ÷ 3 = 8)
- ✅ **1 Aggregator VM** using FedAvg for weight exchange
- ✅ **TCP communication** for reliable weight transfer
- ✅ **Multipass VMs** for isolation and reproducibility
- ✅ **Sender processes** stream UA-DETRAC images to edge VMs
- ✅ **Local training** on each edge before federated aggregation
- ✅ **Metrics collection** for training monitoring
- ✅ **100 rounds** of federated learning (all UA-DETRAC images used)

## How It Works

1. **Senders** stream images from assigned UA-DETRAC intersections to their edge VMs
2. **Edge agents** receive images, train locally for N epochs
3. **Edge agents** upload local weights to aggregator via TCP
4. **Aggregator** performs FedAvg: `global_weights = mean(edge0_weights, edge1_weights, edge2_weights)`
5. **Aggregator** broadcasts updated global weights to all edge VMs
6. **Edge agents** download updated weights, repeat from step 2

## Notes

- TCP is used instead of UDP because weight updates must arrive intact and in order
- Each edge handles 8 intersections (24 ÷ 3 = 8)
- The aggregator doesn't need UA-DETRAC data, only edge VMs do
- All VMs communicate over TCP on configurable ports (default: 5000 for aggregator, 6000-6002 for edge data)
